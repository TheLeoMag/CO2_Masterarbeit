"""Publication-quality figures for the Styrian 100 m population-grid notebook."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import PercentFormatter
from shapely.geometry import MultiPolygon, Point, Polygon, box
from xyzservices import TileProvider

from plotting.apa7 import add_north_arrow, new_figure, save_apa7


CRS = "EPSG:3035"
CELL_SIZE_M = 100
YEARS = tuple(range(2015, 2026))
MAP_WIDTH = 7
MAP_HEIGHT = 5
BASEMAP_AT_GRAY = TileProvider(
    name="basemap.at Grau",
    url="https://maps.wien.gv.at/basemap/bmapgrau/normal/google3857/{z}/{y}/{x}.png",
    attribution="Grundkarte: basemap.at",
    max_zoom=20,
)


def _normalise_id(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def _save_pair(fig, output_dir: Path, stem: str) -> None:
    save_apa7(fig, output_dir / f"{stem}.png")
    save_apa7(fig, output_dir / f"{stem}.svg")


def _style_map_axis(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")


def _add_scale_bar(
    ax,
    length_m: float = 10_000,
    label: str | None = None,
    x_fraction: float = 0.07,
    y_fraction: float = 0.055,
    clip_on: bool = True,
) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + x_fraction * (xmax - xmin)
    y0 = ymin + y_fraction * (ymax - ymin)
    ax.plot(
        [x0, x0 + length_m],
        [y0, y0],
        color="black",
        linewidth=1.5,
        zorder=30,
        clip_on=clip_on,
    )
    ax.plot(
        [x0, x0],
        [y0 - 0.008 * (ymax - ymin), y0 + 0.008 * (ymax - ymin)],
        color="black",
        linewidth=1,
        clip_on=clip_on,
    )
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - 0.008 * (ymax - ymin), y0 + 0.008 * (ymax - ymin)],
        color="black",
        linewidth=1,
        clip_on=clip_on,
    )
    if label is None:
        label = f"{length_m / 1_000:g} km"
    ax.text(
        x0 + length_m / 2,
        y0 + 0.018 * (ymax - ymin),
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        clip_on=clip_on,
    )


def load_inputs(project_dir: Path) -> dict[str, object]:
    project_dir = Path(project_dir)
    population = gpd.read_parquet(
        project_dir / "OGD" / "population_grid_styria.geoparquet"
    ).to_crs(CRS)
    population["population"] = pd.to_numeric(population["population"], errors="raise").astype(float)

    raster_table = pd.read_parquet(
        project_dir / "ANAL" / "data" / "raster_100m_styria.geoparquet",
        columns=[
            "grid_id",
            "easting",
            "northing",
            "municipality_id",
            "municipality_name",
            "population_2025",
        ],
    )
    raster_table["municipality_id"] = _normalise_id(raster_table["municipality_id"])

    firms = pd.read_parquet(
        project_dir / "ANAL" / "data" / "firms_assigned_100m.geoparquet",
        columns=["grid_id_100m"],
    )
    firm_counts = firms.groupby("grid_id_100m").size().rename("firm_count")

    municipalities = gpd.read_file(
        project_dir / "OGD" / "Gemeindegrenzen.zip",
        layer="Gemeindegrenzen",
        columns=["GEMNR6", "GEMNAM"],
    ).to_crs(CRS)
    municipalities = municipalities.rename(
        columns={"GEMNR6": "municipality_id", "GEMNAM": "municipality_name"}
    )
    municipalities["municipality_id"] = _normalise_id(municipalities["municipality_id"])
    dissolved = municipalities.geometry.make_valid().union_all()
    if isinstance(dissolved, Polygon):
        styria_boundary = Polygon(dissolved.exterior)
    elif isinstance(dissolved, MultiPolygon):
        styria_boundary = MultiPolygon([Polygon(part.exterior) for part in dissolved.geoms])
    else:
        raise TypeError(f"Unexpected dissolved Styrian boundary type: {dissolved.geom_type}")

    municipal_population = pd.read_csv(
        project_dir / "OGD" / "STMK_POP_2002_2025.csv",
        sep=";",
        encoding="cp1252",
    )
    municipal_population["municipality_id"] = _normalise_id(municipal_population["LAU_CODE"])

    raster_municipality_2025 = (
        raster_table.groupby("municipality_id", as_index=False)["population_2025"]
        .sum()
        .rename(columns={"population_2025": "raster_population_2025"})
    )
    metrics = municipal_population.merge(
        raster_municipality_2025, on="municipality_id", how="left", validate="one_to_one"
    )
    metrics["growth_2015_2025_pct"] = (metrics["POP_2025"] / metrics["POP_2015"] - 1) * 100
    for year in YEARS:
        metrics[f"scale_{year}"] = metrics[f"POP_{year}"] / metrics["raster_population_2025"]

    return {
        "project_dir": project_dir,
        "population": population,
        "raster_table": raster_table,
        "firm_counts": firm_counts,
        "municipalities": municipalities,
        "styria_boundary": styria_boundary,
        "municipal_metrics": metrics,
    }


def plot_population_overview(data: dict[str, object], output_dir: Path) -> None:
    population = data["population"]
    boundary = data["styria_boundary"]
    norm = LogNorm(vmin=1, vmax=float(population["population"].max()))

    fig, ax = new_figure(MAP_WIDTH, MAP_HEIGHT)
    population.plot(
        ax=ax,
        column="population",
        cmap="Greys",
        norm=norm,
        linewidth=0,
        rasterized=True,
        legend=True,
        legend_kwds={"label": "Bevölkerung je 100-m-Zelle (logarithmisch)", "shrink": 0.70},
    )
    gpd.GeoSeries([boundary], crs=CRS).boundary.plot(ax=ax, color="black", linewidth=0.8)
    _style_map_axis(ax)
    add_north_arrow(ax)
    _add_scale_bar(ax, 25_000, "25 km")
    _save_pair(fig, output_dir, "figure_2_population_grid_overview_log")
    plt.show()


def _projected_point(lon: float, lat: float):
    return gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(CRS).iloc[0]


def _plot_equal_scale_details(
    data: dict[str, object],
    output_dir: Path,
    norm: LogNorm,
    output_stem: str,
    cmap: str = "Greys",
    basemap_source: TileProvider | None = None,
    population_alpha: float = 1.0,
) -> None:
    population = data["population"]
    municipalities = data["municipalities"]
    locations = (
        ("Graz-Innenstadt", _projected_point(15.4395, 47.0707)),
        ("Leoben", _projected_point(15.0940, 47.3765)),
        ("Eibiswald", _projected_point(15.2470, 46.6860)),
    )
    half_width = 3_000
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5))

    for ax, (title, centre) in zip(axes, locations):
        xmin, xmax = centre.x - half_width, centre.x + half_width
        ymin, ymax = centre.y - half_width, centre.y + half_width
        cells = population.cx[xmin:xmax, ymin:ymax]
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        if basemap_source is not None:
            cx.add_basemap(
                ax,
                crs=CRS,
                source=basemap_source,
                zoom=14,
                attribution=False,
                zorder=0,
            )
        cells.plot(
            ax=ax,
            column="population",
            cmap=cmap,
            norm=norm,
            linewidth=0.08,
            edgecolor="0.82",
            alpha=population_alpha,
            rasterized=True,
            zorder=2,
        )
        municipalities.boundary.plot(ax=ax, color="0.25", linewidth=0.45, zorder=3)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title=title)
        _style_map_axis(ax)

    # Shared orientation aids: the scale belongs below the first panel and the
    # north arrow appears only in the third panel to avoid visual repetition.
    _add_scale_bar(
        axes[0],
        1_000,
        "1 km",
        y_fraction=-0.07,
        clip_on=False,
    )
    add_north_arrow(axes[2], x=0.92, y=0.06, length=0.10)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.02, shrink=0.82)
    colorbar.set_label("Bevölkerung je 100-m-Zelle (logarithmisch)")
    _save_pair(fig, output_dir, output_stem)
    plt.show()


def plot_equal_scale_details(data: dict[str, object], output_dir: Path) -> None:
    population = data["population"]
    norm = LogNorm(vmin=1, vmax=float(population["population"].quantile(0.995)))
    _plot_equal_scale_details(
        data,
        output_dir,
        norm,
        "figure_3_population_grid_equal_scale_details",
    )


def plot_equal_scale_details_figure_3a(data: dict[str, object], output_dir: Path) -> None:
    """Render the detail panels with a logarithmic purple-to-green scale."""
    population = data["population"]
    norm = LogNorm(vmin=1, vmax=float(population["population"].max()))
    _plot_equal_scale_details(
        data,
        output_dir,
        norm,
        "figure_3a_population_grid_equal_scale_viridis",
        cmap="viridis",
    )


def plot_equal_scale_details_figure_3b(data: dict[str, object], output_dir: Path) -> None:
    """Render Figure 3a over the official grey basemap.at tile service."""
    population = data["population"]
    norm = LogNorm(vmin=1, vmax=float(population["population"].max()))
    _plot_equal_scale_details(
        data,
        output_dir,
        norm,
        "figure_3b_population_grid_equal_scale_basemap_at",
        cmap="viridis",
        basemap_source=BASEMAP_AT_GRAY,
        population_alpha=0.72,
    )


def plot_concentration_curve(data: dict[str, object], output_dir: Path) -> pd.DataFrame:
    values = np.sort(data["population"]["population"].to_numpy(dtype=float))[::-1]
    cumulative_population = np.cumsum(values) / values.sum()
    cell_share = np.arange(1, len(values) + 1) / len(values)
    thresholds = {}
    for share in (0.50, 0.80, 0.90):
        thresholds[share] = cell_share[np.searchsorted(cumulative_population, share)]

    fig, ax = new_figure(6.5, 4.2)
    ax.plot(cell_share, cumulative_population, color="black", linewidth=1.6)
    ax.plot([0, 1], [0, 1], color="0.65", linewidth=0.8, linestyle="--")
    x50 = thresholds[0.50]
    ax.scatter([x50], [0.50], color="black", s=24, zorder=3)
    ax.annotate(
        f"50 % der Bevölkerung in\n{x50:.1%} der bewohnten Zellen",
        xy=(x50, 0.50),
        xytext=(x50 + 0.12, 0.34),
        arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.8},
        fontsize=8,
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Anteil der bewohnten 100-m-Zellen (absteigend sortiert)",
        ylabel="Kumulierter Bevölkerungsanteil",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save_pair(fig, output_dir, "figure_4_population_concentration_curve")
    plt.show()

    return pd.DataFrame(
        {
            "population_share": list(thresholds),
            "minimum_share_of_populated_cells": list(thresholds.values()),
        }
    )


def build_risk_cells(data: dict[str, object]) -> gpd.GeoDataFrame:
    raster = data["raster_table"]
    firm_counts = data["firm_counts"]
    firm_ids = set(firm_counts.index)
    active = raster.loc[
        raster["population_2025"].gt(0) | raster["grid_id"].isin(firm_ids)
    ].copy()
    active["has_population"] = active["population_2025"].gt(0)
    active["firm_count"] = active["grid_id"].map(firm_counts).fillna(0).astype(int)
    active["has_firm"] = active["firm_count"].gt(0)
    active["risk_category"] = np.select(
        [
            active["has_population"] & active["has_firm"],
            active["has_population"],
            active["has_firm"],
        ],
        ["Schnittmenge", "Nur Bevölkerung", "Nur Unternehmensbestand"],
        default="Nicht relevant",
    )
    geometry = [
        box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M)
        for x, y in zip(active["easting"], active["northing"])
    ]
    return gpd.GeoDataFrame(active, geometry=geometry, crs=CRS)


def plot_risk_set_map(data: dict[str, object], output_dir: Path) -> pd.DataFrame:
    risk_cells = build_risk_cells(data)
    boundary = data["styria_boundary"]
    colors = {
        "Nur Bevölkerung": "0.78",
        "Nur Unternehmensbestand": "#b2182b",
        "Schnittmenge": "0.12",
    }
    order = ("Nur Bevölkerung", "Nur Unternehmensbestand", "Schnittmenge")

    fig, ax = new_figure(MAP_WIDTH, MAP_HEIGHT)
    for category in order:
        subset = risk_cells[risk_cells["risk_category"].eq(category)]
        subset.plot(
            ax=ax,
            color=colors[category],
            linewidth=0,
            rasterized=True,
            zorder=2 if category == "Nur Bevölkerung" else 3,
        )
    gpd.GeoSeries([boundary], crs=CRS).boundary.plot(ax=ax, color="black", linewidth=0.8)
    handles = [
        Patch(
            facecolor=colors[category],
            edgecolor="none",
            label=f"{category} ({risk_cells['risk_category'].eq(category).sum():,})",
        )
        for category in order
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7)
    _style_map_axis(ax)
    add_north_arrow(ax)
    _add_scale_bar(ax, 25_000, "25 km", x_fraction=0.58)
    _save_pair(fig, output_dir, "figure_5_population_firm_risk_set")
    plt.show()

    return (
        risk_cells.groupby("risk_category", as_index=False)
        .size()
        .rename(columns={"size": "cells"})
    )


def plot_raster_vs_municipality(data: dict[str, object], output_dir: Path) -> None:
    population = data["population"].copy()
    municipalities = data["municipalities"].merge(
        data["municipal_metrics"][["municipality_id", "POP_2025"]],
        on="municipality_id",
        how="left",
        validate="one_to_one",
    )
    population["population_density_km2"] = population["population"] * 100
    municipalities["population_density_km2"] = (
        municipalities["POP_2025"] / (municipalities.geometry.area / 1_000_000)
    )
    centre = _projected_point(15.4395, 47.0707)
    half_width = 6_000
    xmin, xmax = centre.x - half_width, centre.x + half_width
    ymin, ymax = centre.y - half_width, centre.y + half_width
    norm = LogNorm(vmin=10, vmax=40_000)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharex=True, sharey=True)

    population.cx[xmin:xmax, ymin:ymax].plot(
        ax=axes[0],
        column="population_density_km2",
        cmap="Greys",
        norm=norm,
        linewidth=0,
        rasterized=True,
    )
    municipalities.cx[xmin:xmax, ymin:ymax].plot(
        ax=axes[1],
        column="population_density_km2",
        cmap="Greys",
        norm=norm,
        linewidth=0.45,
        edgecolor="white",
    )
    for ax, title in zip(axes, ("100-m-Raster", "Gemeindeaggregat")):
        municipalities.boundary.plot(ax=ax, color="0.25", linewidth=0.45)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title=title)
        _style_map_axis(ax)
        _add_scale_bar(ax, 2_000, "2 km")
        add_north_arrow(ax, x=0.92, y=0.06, length=0.10)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="Greys")
    colorbar = fig.colorbar(scalar, ax=axes, fraction=0.03, pad=0.02, shrink=0.82)
    colorbar.set_label("Einwohner je km² (logarithmisch)")
    _save_pair(fig, output_dir, "figure_6_raster_vs_municipality_graz")
    plt.show()


def plot_backcast_workflow(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.4))
    ax.set_axis_off()
    steps = (
        ("Raster 2025", "Bevölkerung je\n100-m-Zelle"),
        ("Gemeindezuordnung", "Zellzentrum →\nGemeinde g"),
        ("Gemeindebevölkerung", "Bevölkerung\nim Jahr t"),
        ("Skalierungsfaktor", "$F_{g,t}=P_{g,t}/P_{g,2025}^{Raster}$"),
        ("Rasterwert Jahr t", "$P_{i,t}=P_{i,2025}\\times F_{g,t}$"),
    )
    lefts = np.linspace(0.015, 0.815, len(steps))
    width, height, y = 0.17, 0.48, 0.28
    for index, ((title, subtitle), left) in enumerate(zip(steps, lefts)):
        face = "0.92" if index in (0, 4) else "white"
        patch = FancyBboxPatch(
            (left, y),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=face,
            edgecolor="black",
            linewidth=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(left + width / 2, y + 0.33, title, ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.text(left + width / 2, y + 0.16, subtitle, ha="center", va="center", fontsize=7, color="0.25", transform=ax.transAxes)
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(lefts[index + 1] - 0.008, y + height / 2),
                xytext=(left + width + 0.008, y + height / 2),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
            )
    ax.text(0.5, 0.10, "Die räumliche Verteilung innerhalb jeder Gemeinde bleibt konstant.", ha="center", va="center", fontsize=8, color="0.30", transform=ax.transAxes)
    _save_pair(fig, output_dir, "figure_7_population_backcast_workflow")
    plt.show()


def plot_indexed_municipal_trends(data: dict[str, object], output_dir: Path) -> None:
    metrics = data["municipal_metrics"]
    indexed = np.column_stack(
        [metrics[f"POP_{year}"].to_numpy(dtype=float) / metrics["POP_2015"].to_numpy(dtype=float) * 100 for year in YEARS]
    )
    q10, median, q90 = np.quantile(indexed, [0.10, 0.50, 0.90], axis=0)
    fig, ax = new_figure(7, 4.6)
    for row in indexed:
        ax.plot(YEARS, row, color="0.72", linewidth=0.45, alpha=0.42, zorder=1)
    ax.fill_between(YEARS, q10, q90, color="0.65", alpha=0.28, linewidth=0, label="10.–90. Perzentil")
    ax.plot(YEARS, median, color="black", linewidth=1.8, label="Median")
    ax.axhline(100, color="0.35", linewidth=0.8, linestyle="--")
    ax.set(xticks=YEARS, xlabel="Jahr", ylabel="Gemeindebevölkerung (2015 = 100)")
    ax.legend(frameon=False, loc="upper left")
    _save_pair(fig, output_dir, "figure_8_indexed_municipal_population")
    plt.show()


def _spread_label_positions(
    values: np.ndarray,
    minimum_gap: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    order = np.argsort(values)
    positions = values[order].astype(float).copy()
    positions[0] = max(positions[0], lower)
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)
    if positions[-1] > upper:
        positions -= positions[-1] - upper
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - minimum_gap)
    result = np.empty_like(positions)
    result[order] = positions
    return result


def plot_indexed_municipal_trends_labelled(data: dict[str, object], output_dir: Path) -> pd.DataFrame:
    metrics = data["municipal_metrics"].copy()
    indexed = np.column_stack(
        [metrics[f"POP_{year}"].to_numpy(dtype=float) / metrics["POP_2015"].to_numpy(dtype=float) * 100 for year in YEARS]
    )
    metrics["change_2015_2025_pct"] = indexed[:, -1] - 100
    selected_indices = pd.concat(
        [
            metrics.nsmallest(5, "change_2015_2025_pct"),
            metrics.nlargest(5, "change_2015_2025_pct"),
        ]
    ).index.to_numpy()
    selected = metrics.loc[selected_indices].copy()
    selected["group"] = np.where(
        selected["change_2015_2025_pct"].ge(0),
        "Stärkste Zunahme",
        "Stärkste Abnahme",
    )
    fig, ax = new_figure(7.8, 4.8)
    for row in indexed:
        ax.plot(YEARS, row, color="0.80", linewidth=0.40, alpha=0.45, zorder=1)
    ax.axhline(100, color="0.40", linewidth=0.8, linestyle="--", zorder=2)

    lower = float(np.nanmin(indexed))
    upper = float(np.nanmax(indexed))
    padding = max((upper - lower) * 0.04, 1.5)
    ax.set_ylim(lower - padding, upper + padding)
    label_y = _spread_label_positions(
        indexed[selected_indices, -1],
        minimum_gap=max((upper - lower) * 0.036, 1.7),
        lower=lower - padding * 0.4,
        upper=upper + padding * 0.4,
    )

    for row_index, municipality_index in enumerate(selected_indices):
        values = indexed[municipality_index]
        ax.plot(YEARS, values, color="black", linewidth=1.35, zorder=3)
        ax.annotate(
            f"{selected.loc[municipality_index, 'LAU_NAME']} "
            f"({selected.loc[municipality_index, 'change_2015_2025_pct']:+.1f} %)",
            xy=(YEARS[-1], values[-1]),
            xytext=(1.025, label_y[row_index]),
            xycoords="data",
            textcoords=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            color="black",
            annotation_clip=False,
        )

    ax.set(xticks=YEARS, xlim=(YEARS[0], YEARS[-1]), xlabel="Jahr", ylabel="Gemeindebevölkerung (2015 = 100)")
    _save_pair(fig, output_dir, "figure_8b_indexed_municipal_population_labelled")
    plt.show()
    return selected[["LAU_CODE", "LAU_NAME", "change_2015_2025_pct", "group"]].sort_values(
        "change_2015_2025_pct", ascending=False
    )


def plot_scaling_factor_boxplots(data: dict[str, object], output_dir: Path) -> None:
    metrics = data["municipal_metrics"]
    values = [metrics[f"scale_{year}"].dropna().to_numpy() for year in YEARS]
    fig, ax = new_figure(7, 4.6)
    ax.boxplot(
        values,
        tick_labels=YEARS,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "0.78", "edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    ax.axhline(1, color="0.35", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Jahr", ylabel="Gemeindespezifischer Skalierungsfaktor")
    _save_pair(fig, output_dir, "figure_9_population_scaling_factors")
    plt.show()


def plot_municipal_growth_map(data: dict[str, object], output_dir: Path) -> None:
    municipalities = data["municipalities"].merge(
        data["municipal_metrics"][["municipality_id", "growth_2015_2025_pct"]],
        on="municipality_id",
        how="left",
        validate="one_to_one",
    )
    limit = float(np.nanquantile(np.abs(municipalities["growth_2015_2025_pct"]), 0.98))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = new_figure(MAP_WIDTH, MAP_HEIGHT)
    municipalities.plot(
        ax=ax,
        column="growth_2015_2025_pct",
        cmap="RdBu_r",
        norm=norm,
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Bevölkerungsveränderung 2015–2025 (%)", "shrink": 0.70},
    )
    municipalities.boundary.plot(ax=ax, color="0.25", linewidth=0.25)
    _style_map_axis(ax)
    add_north_arrow(ax)
    _add_scale_bar(ax, 25_000, "25 km")
    _save_pair(fig, output_dir, "figure_10_municipal_population_growth_2015_2025")
    plt.show()
