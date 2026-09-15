"""Shared data preparation and diagnostics for the thesis model notebooks.

The notebooks deliberately keep estimation and presentation separate.  This
module contains only deterministic preparation code so the assumption check
and estimation notebooks use exactly the same samples and transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


START_YEAR = 2016
END_YEAR = 2025
LAG_YEAR = START_YEAR - 1
CAR_CONTOURS = [15]
WALK_CONTOURS = [10]
SURVIVAL_BAND = "0_15"


@dataclass(frozen=True)
class ProjectPaths:
    project: Path
    data: Path
    features: Path
    panel: Path
    firms: Path
    births_by_fachgruppe: Path
    models: Path
    cache: Path


def discover_project_dir(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or repository working dir."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "ANAL").is_dir() and (candidate / "OGD").is_dir():
            return candidate
    raise RuntimeError("Run from the project directory or one of its subdirectories.")


def project_paths(start: Path | None = None) -> ProjectPaths:
    project = discover_project_dir(start)
    data = project / "ANAL" / "data"
    return ProjectPaths(
        project=project,
        data=data,
        features=data / "routing" / "features",
        panel=data / "raster_quarter_panel_100m.parquet",
        firms=data / "firms_assigned_100m.geoparquet",
        births_by_fachgruppe=data / "births_by_fachgruppe_100m.parquet",
        models=data / "models",
        cache=data / "cache",
    )


def _require_columns(requirements: dict[Path, set[str]]) -> None:
    missing_files = [path for path in requirements if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required model inputs: {missing_files}")
    for path, required in requirements.items():
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")


def preflight_founding(paths: ProjectPaths) -> None:
    panel_required = {
        "grid_id", "municipality_id", "year", "quarter", "period",
        "births", "active_firms_tminus1",
    }
    car_required = {
        "grid_id", "year", "quarter", "period", "own_cell_pop", "own_cell_firms",
        *{f"pop_access_{m}min" for m in CAR_CONTOURS},
        *{f"existing_firms_access_{m}min" for m in CAR_CONTOURS},
    }
    walk_required = {
        "grid_id", "year", "quarter", "period", "own_cell_walk_pop",
        "own_cell_walk_firms", "pt_ohne_haltestelle",
        *{f"walk_pop_{m}min" for m in WALK_CONTOURS},
        *{f"walk_firms_{m}min" for m in WALK_CONTOURS},
        *{f"walk_pt_routes_{m}min" for m in WALK_CONTOURS},
    }
    nearest_required = {"grid_id", "year", "tt_motorway_exit_min"}
    requirements = {paths.panel: panel_required}
    for year in range(LAG_YEAR, END_YEAR + 1):
        year_dir = paths.features / str(year)
        requirements[year_dir / "accessibility_potentials_100m.parquet"] = car_required
        requirements[year_dir / "pedestrian_accessibility_quarter_100m.parquet"] = walk_required
        requirements[year_dir / "nearest_infrastructure_100m.parquet"] = nearest_required
    _require_columns(requirements)


def preflight_survival(paths: ProjectPaths) -> None:
    firm_required = {
        "firm_id", "founding_date", "exit_date", "exit_observed",
        "grid_id_100m", "Fachgruppe_ID", "Sparte_ID", "Sparte_Text",
    }
    access_required = {
        "firm_id", "grid_id_100m", "Fachgruppe_ID", "year", "quarter", "period",
        "included_in_lagged_stock", "own_cell_pop", "own_cell_firms",
        "own_cell_same_fachgruppe_firms",
        *{f"pop_access_{m}min" for m in CAR_CONTOURS},
        *{f"existing_firms_access_{m}min" for m in CAR_CONTOURS},
        *{f"same_fachgruppe_firms_access_{m}min" for m in CAR_CONTOURS},
    }
    walk_required = {
        "grid_id", "year", "quarter", "walk_pt_routes_10min", "pt_ohne_haltestelle",
    }
    nearest_required = {"grid_id", "year", "tt_motorway_exit_min"}
    requirements = {paths.firms: firm_required}
    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = paths.features / str(year)
        requirements[year_dir / "firm_accessibility_quarter_100m.parquet"] = access_required
        requirements[year_dir / "pedestrian_accessibility_quarter_100m.parquet"] = walk_required
        requirements[year_dir / "nearest_infrastructure_100m.parquet"] = nearest_required
    _require_columns(requirements)


def _feature_glob(paths: ProjectPaths, filename: str) -> str:
    return (paths.features / "*" / filename).as_posix()


def _build_rings(
    frame: pd.DataFrame, prefix: str, contours: Iterable[int], own_column: str
) -> pd.DataFrame:
    contours = list(contours)
    own = frame[own_column].to_numpy(dtype="float64")
    cumulative = {
        m: np.clip(frame[f"{prefix}_{m}min"].to_numpy(dtype="float64") - own, 0, None)
        for m in contours
    }
    rings: dict[str, np.ndarray] = {}
    lower = 0
    for upper in contours:
        previous = 0.0 if lower == 0 else cumulative[lower]
        rings[f"{prefix}_ring_{lower}_{upper}"] = np.clip(cumulative[upper] - previous, 0, None)
        lower = upper
    return pd.DataFrame(rings, index=frame.index)


def founding_model_data(
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[tuple[str, str, str]]]:
    """Return the founding analysis frame, X, y, clusters, and model specification."""
    preflight_founding(paths)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    query = f"""
        SELECT p.grid_id, p.municipality_id, p.year, p.quarter, p.period,
               p.births, p.active_firms_tminus1,
               a.own_cell_pop, a.own_cell_firms,
               w.own_cell_walk_pop, w.own_cell_walk_firms,
               a.pop_access_15min, a.existing_firms_access_15min,
               w.walk_pop_10min, w.walk_firms_10min,
               w.walk_pt_routes_10min, w.pt_ohne_haltestelle,
               n.tt_motorway_exit_min
        FROM read_parquet('{paths.panel.as_posix()}') p
        JOIN read_parquet('{_feature_glob(paths, 'accessibility_potentials_100m.parquet')}') a
          USING (grid_id, year, quarter, period)
        JOIN read_parquet('{_feature_glob(paths, 'pedestrian_accessibility_quarter_100m.parquet')}') w
          USING (grid_id, year, quarter, period)
        JOIN read_parquet('{_feature_glob(paths, 'nearest_infrastructure_100m.parquet')}') n
          USING (grid_id, year)
        WHERE p.year BETWEEN {LAG_YEAR} AND {END_YEAR}
        ORDER BY p.grid_id, p.year, p.quarter
    """
    frame = con.execute(query).df()
    con.close()
    if frame.duplicated(["grid_id", "year", "quarter"]).any():
        raise ValueError("The founding join duplicated cell-quarter rows.")

    routed = [
        "pop_access_15min", "existing_firms_access_15min", "walk_pop_10min",
        "walk_firms_10min", "walk_pt_routes_10min", "pt_ohne_haltestelle",
        "tt_motorway_exit_min",
    ]
    frame = frame.loc[~frame[routed].isna().any(axis=1)].copy()
    rings = pd.concat(
        [
            _build_rings(frame, "pop_access", CAR_CONTOURS, "own_cell_pop"),
            _build_rings(frame, "existing_firms_access", CAR_CONTOURS, "own_cell_firms"),
            _build_rings(frame, "walk_pop", WALK_CONTOURS, "own_cell_walk_pop"),
            _build_rings(frame, "walk_firms", WALK_CONTOURS, "own_cell_walk_firms"),
        ],
        axis=1,
    )
    frame = pd.concat([frame, rings], axis=1)
    for column in rings:
        frame[f"log_{column}"] = np.log1p(frame[column])

    ring_specs = [
        ("log_pop_access_ring_0_15", "log_existing_firms_access_ring_0_15", "log_firms_relative_car_ring_0_15"),
        ("log_walk_pop_ring_0_10", "log_walk_firms_ring_0_10", "log_firms_relative_walk_ring_0_10"),
    ]
    mass_columns: list[str] = []
    relative_density_columns: list[str] = []
    for mass, stock, relative_density in ring_specs:
        # This is log((1 + firms) / (1 + population)), not a literal per-capita rate.
        frame[relative_density] = frame[stock] - frame[mass]
        mass_columns.append(mass)
        relative_density_columns.append(relative_density)

    frame = frame.sort_values(["grid_id", "year", "quarter"])
    period_number = frame["year"] * 4 + frame["quarter"]
    previous_period = period_number.groupby(frame["grid_id"], sort=False).shift(4)
    previous_population = frame.groupby("grid_id", sort=False)["own_cell_pop"].shift(4)
    previous_population = previous_population.where(period_number - previous_period == 4)
    frame["log_own_firms"] = np.log1p(frame["active_firms_tminus1"])
    frame["log_own_pop"] = np.log1p(previous_population)
    frame["population_growth_yoy"] = np.log1p(frame["own_cell_pop"]) - np.log1p(previous_population)
    frame["log_tt_motorway_exit"] = np.log1p(frame["tt_motorway_exit_min"])
    frame = frame.loc[frame["year"] >= START_YEAR].copy()

    regressors = [
        "log_own_firms", "log_own_pop", *mass_columns, *relative_density_columns,
        "population_growth_yoy", "log_tt_motorway_exit",
        "walk_pt_routes_10min", "pt_ohne_haltestelle",
    ]
    frame = frame.dropna(subset=regressors).copy()
    dummies = pd.get_dummies(frame["period"], prefix="period", drop_first=True, dtype="float64")
    design = pd.concat([frame[regressors].astype("float64"), dummies], axis=1)
    design.insert(0, "const", 1.0)
    outcome = frame["births"].astype("float64")
    clusters = frame["grid_id"]
    if not np.isfinite(design.to_numpy()).all() or (outcome < 0).any():
        raise ValueError("Invalid founding design matrix or outcome.")
    if not np.allclose(outcome, np.floor(outcome)):
        raise ValueError("Founding outcome must contain integer counts.")
    return frame, design, outcome, clusters, regressors, ring_specs


def _add_log_rings(
    frame: pd.DataFrame, source, prefix: str, contours: Iterable[int], own_column: str
) -> list[str]:
    contours = list(contours)
    own = frame[own_column].to_numpy(dtype="float64")
    cumulative = {
        m: np.clip(frame[source(m)].to_numpy(dtype="float64") - own, 0, None)
        for m in contours
    }
    columns: list[str] = []
    lower = 0
    for upper in contours:
        previous = 0.0 if lower == 0 else cumulative[lower]
        column = f"log_{prefix}_ring_{lower}_{upper}"
        frame[column] = np.log1p(np.clip(cumulative[upper] - previous, 0, None))
        columns.append(column)
        lower = upper
    return columns


def survival_model_data(
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return interval-level survival data and the exact PHReg model frame."""
    preflight_survival(paths)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    firm_glob = _feature_glob(paths, "firm_accessibility_quarter_100m.parquet")
    spells = con.execute(
        f"""
        SELECT a.firm_id AS standort_id, a.grid_id_100m AS grid_id,
               CAST(a.Fachgruppe_ID AS VARCHAR) AS Fachgruppe_ID,
               a.year, a.quarter, a.period, a.included_in_lagged_stock,
               a.own_cell_pop, a.own_cell_firms, a.own_cell_same_fachgruppe_firms,
               a.pop_access_15min, a.existing_firms_access_15min,
               a.same_fachgruppe_firms_access_15min,
               w.walk_pt_routes_10min, w.pt_ohne_haltestelle,
               n.tt_motorway_exit_min
        FROM read_parquet('{firm_glob}') a
        JOIN read_parquet('{_feature_glob(paths, 'pedestrian_accessibility_quarter_100m.parquet')}') w
          ON a.grid_id_100m = w.grid_id AND a.year = w.year AND a.quarter = w.quarter
        JOIN read_parquet('{_feature_glob(paths, 'nearest_infrastructure_100m.parquet')}') n
          ON a.grid_id_100m = n.grid_id AND a.year = n.year
        WHERE a.year BETWEEN {START_YEAR} AND {END_YEAR}
        ORDER BY a.firm_id, a.year, a.quarter
        """
    ).df()
    con.close()
    if spells.duplicated(["standort_id", "year", "quarter"]).any():
        raise ValueError("The survival join duplicated firm-quarter rows.")

    routed = [
        "pop_access_15min", "existing_firms_access_15min",
        "same_fachgruppe_firms_access_15min", "walk_pt_routes_10min",
        "pt_ohne_haltestelle", "tt_motorway_exit_min",
    ]
    incomplete_ids = spells.loc[spells[routed].isna().any(axis=1), "standort_id"].unique()
    missing_group_ids = spells.loc[spells["Fachgruppe_ID"].isna(), "standort_id"].unique()
    excluded = np.union1d(incomplete_ids, missing_group_ids)
    spells = spells.loc[~spells["standort_id"].isin(excluded)].copy()

    firms = pd.read_parquet(
        paths.firms,
        columns=[
            "firm_id", "founding_date", "exit_date", "exit_observed",
            "Sparte_ID", "Sparte_Text",
        ],
    ).rename(columns={"firm_id": "standort_id", "Sparte_ID": "sparte", "Sparte_Text": "sparte_name"})
    if firms["standort_id"].duplicated().any():
        raise ValueError("firm_id/standort_id is not unique in the firm source.")
    firms["founding_date"] = pd.to_datetime(firms["founding_date"], errors="coerce")
    firms["exit_date"] = pd.to_datetime(firms["exit_date"], errors="coerce")
    firms["sparte"] = firms["sparte"].astype("string")
    firms["sparte_name"] = firms["sparte_name"].astype("string")
    spells = spells.merge(firms, on="standort_id", how="left", validate="many_to_one")
    if spells[["founding_date", "sparte", "sparte_name"]].isna().any().any():
        raise ValueError("A routed firm lacks a founding date or official Sparte mapping.")

    period_index = pd.PeriodIndex(spells["period"], freq="Q").asi8
    founding_index = spells["founding_date"].dt.to_period("Q").array.asi8
    spells["start"] = period_index - founding_index
    spells["stop"] = spells["start"] + 1
    exit_period = spells["exit_date"].dt.to_period("Q").astype("string")
    spells["event"] = (
        spells["exit_observed"].fillna(False) & spells["period"].eq(exit_period)
    ).astype("int8")
    spells = spells.sort_values(["standort_id", "year", "quarter"])
    if (spells["start"] < 0).any() or (spells["stop"] <= spells["start"]).any():
        raise ValueError("Invalid survival interval boundaries.")
    if (spells.groupby("standort_id")["event"].sum() > 1).any():
        raise ValueError("A firm has more than one observed exit.")
    period_number = spells["year"] * 4 + spells["quarter"]
    if period_number.groupby(spells["standort_id"]).diff().dropna().ne(1).any():
        raise ValueError("Firm histories contain gaps after routing joins.")

    focal = spells["included_in_lagged_stock"].astype("float64")
    spells["own_firms"] = (spells["own_cell_firms"] - focal).clip(lower=0)
    spells["own_same"] = (spells["own_cell_same_fachgruppe_firms"] - focal).clip(lower=0)
    spells["own_other"] = (spells["own_firms"] - spells["own_same"]).clip(lower=0)
    spells["firms_15"] = (spells["existing_firms_access_15min"] - focal).clip(lower=0)
    spells["same_15"] = (spells["same_fachgruppe_firms_access_15min"] - focal).clip(lower=0)
    spells["other_15"] = (spells["firms_15"] - spells["same_15"]).clip(lower=0)

    _add_log_rings(spells, lambda m: f"pop_access_{m}min", "pop", CAR_CONTOURS, "own_cell_pop")
    _add_log_rings(spells, lambda m: f"same_{m}", "same", CAR_CONTOURS, "own_same")
    _add_log_rings(spells, lambda m: f"other_{m}", "other", CAR_CONTOURS, "own_other")
    spells["log_same_relative_ring_0_15"] = spells["log_same_ring_0_15"] - spells["log_pop_ring_0_15"]
    spells["log_other_relative_ring_0_15"] = spells["log_other_ring_0_15"] - spells["log_pop_ring_0_15"]
    spells["log_own_pop"] = np.log1p(spells["own_cell_pop"])
    spells["log_own_same"] = np.log1p(spells["own_same"])
    spells["log_own_other"] = np.log1p(spells["own_other"])
    spells["log_tt_motorway_exit"] = np.log1p(spells["tt_motorway_exit_min"])
    spells["calendar_year"] = spells["year"] - START_YEAR

    covariates = [
        "log_own_pop", "log_own_same", "log_own_other", "log_pop_ring_0_15",
        "log_same_relative_ring_0_15", "log_other_relative_ring_0_15",
        "log_tt_motorway_exit", "walk_pt_routes_10min", "pt_ohne_haltestelle",
        "calendar_year",
    ]
    model_columns = [
        "standort_id", "grid_id", "Fachgruppe_ID", "sparte", "sparte_name",
        "year", "period", "start", "stop", "event", *covariates,
    ]
    model_frame = spells[model_columns].copy()
    if not np.isfinite(model_frame[covariates].to_numpy(dtype="float64")).all():
        raise ValueError("Invalid survival covariates.")
    return spells, model_frame, covariates


def high_correlations(frame: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    correlations = frame[columns].corr()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
    result = upper[upper.abs() >= threshold].sort_values(key=abs, ascending=False)
    return result.rename("correlation").rename_axis(["term_1", "term_2"]).reset_index()


def official_sector_mapping(paths: ProjectPaths) -> pd.DataFrame:
    """Return the source-provided Fachgruppe-to-Sparte mapping."""
    mapping = (
        pd.read_parquet(paths.firms, columns=["Fachgruppe_ID", "Sparte_ID", "Sparte_Text"])
        .dropna()
        .astype({"Fachgruppe_ID": "string", "Sparte_ID": "string", "Sparte_Text": "string"})
        .drop_duplicates()
        .rename(columns={"Sparte_ID": "sparte", "Sparte_Text": "sparte_name"})
    )
    ambiguous = mapping.groupby("Fachgruppe_ID")["sparte"].nunique()
    if (ambiguous > 1).any():
        raise ValueError(f"Fachgruppen assigned to multiple sectors: {list(ambiguous[ambiguous > 1].index)}")
    sector_names = mapping.groupby("sparte")["sparte_name"].nunique()
    if (sector_names > 1).any():
        raise ValueError(f"Sparte IDs with inconsistent names: {list(sector_names[sector_names > 1].index)}")
    return mapping


def prepare_founding_sector_cache(
    paths: ProjectPaths, *, rebuild: bool = False
) -> tuple[Path, Path, Path, pd.DataFrame]:
    """Aggregate Fachgruppe products to official sectors for optional H4 models."""
    mapping = official_sector_mapping(paths)
    paths.cache.mkdir(parents=True, exist_ok=True)
    access_path = paths.cache / "sparten_accessibility_100m.parquet"
    births_path = paths.cache / "sparten_births_100m.parquet"
    stock_path = paths.cache / "sparten_stock_100m.parquet"
    yearly_dirs = [
        paths.features / str(year) / "fachgruppe_accessibility_quarter_100m.parquet"
        for year in range(LAG_YEAR, END_YEAR + 1)
    ]
    missing = [path for path in [paths.births_by_fachgruppe, *yearly_dirs] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Sector models require missing inputs: {missing}")

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("sector_mapping", mapping[["Fachgruppe_ID", "sparte"]])
    fachgruppe_glob = (
        paths.features / "*" / "fachgruppe_accessibility_quarter_100m.parquet" / "**" / "*.parquet"
    ).as_posix()
    if rebuild or not access_path.exists():
        con.execute(
            f"""
            COPY (
                SELECT f.grid_id, f.year, f.quarter, m.sparte,
                       sum(f.same_fachgruppe_firms_access_15min) AS same_access_15min,
                       sum(f.own_cell_same_fachgruppe_firms) AS own_same
                FROM read_parquet('{fachgruppe_glob}', hive_partitioning = true) f
                JOIN sector_mapping m
                  ON CAST(f.Fachgruppe_ID AS VARCHAR) = m.Fachgruppe_ID
                WHERE f.year BETWEEN {LAG_YEAR} AND {END_YEAR}
                GROUP BY 1, 2, 3, 4
            ) TO '{access_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    if rebuild or not births_path.exists():
        con.execute(
            f"""
            COPY (
                SELECT b.grid_id, b.period, m.sparte, sum(b.births) AS births_sparte
                FROM read_parquet('{paths.births_by_fachgruppe.as_posix()}') b
                JOIN sector_mapping m
                  ON CAST(b.Fachgruppe_ID AS VARCHAR) = m.Fachgruppe_ID
                GROUP BY 1, 2, 3
            ) TO '{births_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    con.close()

    if rebuild or not stock_path.exists():
        panel_columns = set(pq.ParquetFile(paths.panel).schema_arrow.names)
        stock_columns = sorted(
            column for column in panel_columns
            if column.startswith("fachgruppe_") and column.endswith("_active_firms_tminus1")
        )
        keys = pd.read_parquet(paths.panel, columns=["grid_id", "period"])
        pieces: list[pd.DataFrame] = []
        for sector in sorted(mapping["sparte"].unique()):
            fachgruppen = set(mapping.loc[mapping["sparte"] == sector, "Fachgruppe_ID"])
            columns = [
                column for column in stock_columns
                if column.removeprefix("fachgruppe_").removesuffix("_active_firms_tminus1") in fachgruppen
            ]
            if not columns:
                continue
            piece = keys.copy()
            piece["sparte"] = sector
            piece["own_same_stock"] = (
                pd.read_parquet(paths.panel, columns=columns).sum(axis=1).astype("int32").to_numpy()
            )
            pieces.append(piece)
        if not pieces:
            raise ValueError("No sector-specific lagged stock columns were found in the panel.")
        pd.concat(pieces, ignore_index=True).to_parquet(stock_path, index=False)
    return access_path, births_path, stock_path, mapping


def founding_sector_design(
    founding_frame: pd.DataFrame,
    access_path: Path,
    births_path: Path,
    stock_path: Path,
    sector: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Build a sector-specific founding design on the main model's risk set."""
    basis_columns = [
        "grid_id", "year", "quarter", "period", "active_firms_tminus1",
        "own_cell_pop", "own_cell_firms", "log_pop_access_ring_0_15",
        "log_walk_pop_ring_0_10", "population_growth_yoy", "log_own_pop",
        "log_tt_motorway_exit", "walk_pt_routes_10min", "pt_ohne_haltestelle",
        "existing_firms_access_15min",
    ]
    frame = founding_frame[basis_columns].copy()
    access = pd.read_parquet(access_path, filters=[("sparte", "=", sector)]).drop(columns="sparte")
    births = pd.read_parquet(births_path, filters=[("sparte", "=", sector)]).drop(columns="sparte")
    stock = pd.read_parquet(stock_path, filters=[("sparte", "=", sector)]).drop(columns="sparte")
    frame = frame.merge(access, on=["grid_id", "year", "quarter"], how="left", validate="one_to_one")
    frame = frame.merge(stock, on=["grid_id", "period"], how="left", validate="one_to_one")
    frame = frame.merge(births, on=["grid_id", "period"], how="left", validate="one_to_one")
    for column in ["same_access_15min", "own_same", "own_same_stock", "births_sparte"]:
        frame[column] = frame[column].fillna(0)

    frame["own_same_firms"] = frame["own_same_stock"].clip(lower=0)
    frame["own_other_firms"] = (frame["active_firms_tminus1"] - frame["own_same_firms"]).clip(lower=0)
    frame["log_own_same"] = np.log1p(frame["own_same_firms"])
    frame["log_own_other"] = np.log1p(frame["own_other_firms"])
    same_ring = (frame["same_access_15min"] - frame["own_same"]).clip(lower=0)
    all_ring = (frame["existing_firms_access_15min"] - frame["own_cell_firms"]).clip(lower=0)
    other_ring = (all_ring - same_ring).clip(lower=0)
    frame["log_same_relative_car_ring_0_15"] = np.log1p(same_ring) - frame["log_pop_access_ring_0_15"]
    frame["log_other_relative_car_ring_0_15"] = np.log1p(other_ring) - frame["log_pop_access_ring_0_15"]
    regressors = [
        "log_own_same", "log_own_other", "log_own_pop",
        "log_pop_access_ring_0_15", "log_walk_pop_ring_0_10",
        "log_same_relative_car_ring_0_15", "log_other_relative_car_ring_0_15",
        "population_growth_yoy", "log_tt_motorway_exit",
        "walk_pt_routes_10min", "pt_ohne_haltestelle",
    ]
    frame = frame.dropna(subset=regressors)
    dummies = pd.get_dummies(frame["period"], prefix="period", drop_first=True, dtype="float64")
    design = pd.concat([frame[regressors].astype("float64"), dummies], axis=1)
    design.insert(0, "const", 1.0)
    return design, frame["births_sparte"].astype("float64"), frame["grid_id"], regressors


def standardized_condition_number(frame: pd.DataFrame, columns: list[str]) -> float:
    correlation = frame[columns].corr().to_numpy(dtype="float64")
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues[0] <= 0:
        return float("inf")
    return float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))


def cluster_covariance(model, params: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    """Cell-clustered sandwich covariance for a fitted statsmodels PHReg."""
    scores = model.score_residuals(params)
    if np.isinf(scores).any():
        raise ValueError("Infinite PHReg score residuals.")
    outside_risk_sets = np.isnan(scores).any(axis=1)
    scores = np.nan_to_num(scores, nan=0.0)
    codes, labels = pd.factorize(groups, sort=False)
    if (codes < 0).any():
        raise ValueError("Missing cluster labels.")
    grouped = np.zeros((len(labels), scores.shape[1]))
    np.add.at(grouped, codes, scores)
    bread = np.linalg.inv(model.hessian(params))
    covariance = bread @ (grouped.T @ grouped) @ bread
    return covariance, int(outside_risk_sets.sum())
