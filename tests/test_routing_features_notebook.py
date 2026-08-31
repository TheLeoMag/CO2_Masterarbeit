from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gc
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from shapely.geometry import Point, box, shape
from tqdm.auto import tqdm

from ANAL.routing.routing_utils import (
    ACCESS_MINUTES,
    fachgruppe_access_columns,
    fachgruppe_ids,
    fachgruppe_stock_columns,
    main_access_columns,
)
from ANAL.routing.destination_builders import PT_YEAR_SOURCES, pt_stop_destinations


NOTEBOOK_PATH = Path(__file__).parents[1] / "ANAL" / "routing" / "06_generate_features.ipynb"


def notebook_namespace(fachgruppe_ids_for_test: list[str] | None = None) -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = dict(globals())
    namespace.update({
        "FACHGRUPPE_IDS": fachgruppe_ids_for_test or ["101", "102"],
        "START_YEAR": 2015,
        "END_YEAR": 2025,
        "ROUTING_DATA": Path("."),
        "FEATURE_ROOT": Path("."),
        "POPULATION_ACCESS_CONTOURS_MIN": list(ACCESS_MINUTES),
        "PEDESTRIAN_ACCESS_CONTOURS_MIN": [5, 10],
        "NEAREST_INFRA_DESTINATION_STEP_SHARE": 0.10,
        "DESTINATION_CHUNK_SIZE": 100,
    })
    for cell_index in (3, 7, 8):
        source = "".join(notebook["cells"][cell_index]["source"])
        exec(compile(source, f"notebook-cell-{cell_index}", "exec"), namespace)
    orchestration = "".join(notebook["cells"][11]["source"])
    exec(compile(orchestration.split("if RUN_MODE not in")[0], "notebook-orchestration-definitions", "exec"), namespace)
    return namespace


def accessibility_frame(namespace: dict, year: int = 2015) -> pd.DataFrame:
    ids = namespace["FACHGRUPPE_IDS"]
    rows = []
    for grid_id in ("g1", "g2"):
        for quarter in (1, 2, 3, 4):
            row = {
                "grid_id": grid_id,
                "year": year,
                "quarter": quarter,
                "period": f"{year}Q{quarter}",
                "own_cell_pop": 1.0,
                "own_cell_firms": 2.0,
                "own_cell_walk_pop": 1.0,
                "own_cell_walk_firms": 2.0,
                "pt_ohne_haltestelle": 0,
                "created_at": "test",
            }
            for fachgruppe_id in ids:
                row[f"own_cell_fachgruppe_{fachgruppe_id}_firms"] = 1.0
                row[f"own_cell_walk_fachgruppe_{fachgruppe_id}_firms"] = 1.0
            for minutes in ACCESS_MINUTES:
                row[f"pop_access_{minutes}min"] = 10.0
                row[f"existing_firms_access_{minutes}min"] = 5.0
                row[f"reachable_cells_{minutes}min"] = 4
                for fachgruppe_id in ids:
                    row[f"fachgruppe_{fachgruppe_id}_access_{minutes}min"] = 3.0
            for minutes in (5, 10):
                row[f"walk_pop_{minutes}min"] = 8.0
                row[f"walk_firms_{minutes}min"] = 4.0
                row[f"walk_pt_stops_{minutes}min"] = 2
                row[f"walk_pt_departures_{minutes}min"] = 20.0
                row[f"walk_pt_routes_{minutes}min"] = 3
                row[f"reachable_cells_walk_{minutes}min"] = 3
                for fachgruppe_id in ids:
                    row[f"walk_fachgruppe_{fachgruppe_id}_firms_{minutes}min"] = 2.0
            rows.append(row)
    return pd.DataFrame(rows)


def empty_pt_destinations() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "parent_station_id": pd.Series(dtype="object"),
            "pt_departures_weekday": pd.Series(dtype="float64"),
            "route_ids": pd.Series(dtype="object"),
        },
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )


def test_station_named_stops_remain_in_generic_public_transport_pois() -> None:
    stops = gpd.GeoDataFrame(
        {
            "station_id": [1, 2, 3, 4, 5],
            "station_name": ["Graz Hbf", "Leoben Bahnhof", "Lendorf Bahnhst", "Graz WKO", "Graz Busbahnhof"],
            "weekday_avg_departures": [100.0, 80.0, 30.0, 20.0, 50.0],
            "weekday_route_ids": ["S1", "R9", "R5", "40", "X20"],
        },
        geometry=[Point(15.4, 47.1)] * 5,
        crs="EPSG:4326",
    )

    destinations = pt_stop_destinations(stops)

    assert destinations["station_id"].tolist() == [1, 2, 3, 4, 5]
    assert set(destinations["poi_type"]) == {"pt_stop"}


def test_incomplete_2016_transport_source_uses_2017_proxy() -> None:
    assert PT_YEAR_SOURCES[2015] == (2017,)
    assert PT_YEAR_SOURCES[2016] == (2017,)
    assert PT_YEAR_SOURCES[2017] == (2017,)


def test_fachgruppe_ids_are_strings(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    pd.DataFrame({
        "fachgruppe_101_active_firms_tminus1": [1.0],
        "fachgruppe_205_active_firms_tminus1": [2.0],
    }).to_parquet(path, index=False)
    result = fachgruppe_ids(path)
    assert result == ["101", "205"]
    assert all(isinstance(value, str) for value in result)


def test_own_cell_masses_are_separate_and_accessibility_stays_cumulative() -> None:
    namespace = notebook_namespace()
    mass_rows = []
    for grid_id, population, firms, fg101, fg102 in (
        ("g1", 10.0, 3.0, 1.0, 2.0),
        ("g2", 20.0, 4.0, 3.0, 1.0),
    ):
        mass_rows.append({
            "grid_id": grid_id,
            "population_backcast": population,
            "active_firms_tminus1": firms,
            "fachgruppe_101_active_firms_tminus1": fg101,
            "fachgruppe_102_active_firms_tminus1": fg102,
        })
    mass = pd.DataFrame(mass_rows).set_index("grid_id")
    mass_columns = namespace["accessibility_mass_source_columns"]()
    mass_matrix = np.ascontiguousarray(mass[mass_columns].to_numpy(dtype=np.float64))
    mass_matrix.setflags(write=False)
    zero_mass = np.zeros(len(mass_columns), dtype=np.float64)
    zero_mass.setflags(write=False)
    destinations = gpd.GeoDataFrame(
        {"grid_id": ["g1", "g2"], "geometry": [Point(0, 0), Point(1, 0)]},
        crs="EPSG:4326",
    )
    namespace["request_accessibility_isochrones"] = lambda origin, costing, minutes: {
        minute: box(-1, -1, 2, 1) for minute in minutes
    }
    record = namespace["accessibility_records_for_origin"](
        ("g1", 0.0, 0.0),
        2015,
        {1: mass_matrix},
        {1: "2015Q1"},
        destinations,
        {"g1": 0, "g2": 1},
        destinations,
        {"g1": 0, "g2": 1},
        empty_pt_destinations(),
        zero_mass,
        namespace["own_cell_mass_output_columns"](),
        namespace["own_cell_walk_mass_output_columns"](),
        {minutes: namespace["contour_mass_output_columns"](minutes) for minutes in ACCESS_MINUTES},
        {minutes: namespace["pedestrian_contour_mass_output_columns"](minutes) for minutes in (5, 10)},
    )[0]
    assert record["own_cell_pop"] == 10.0
    assert record["own_cell_firms"] == 3.0
    assert record["own_cell_fachgruppe_101_firms"] == 1.0
    assert record["own_cell_fachgruppe_102_firms"] == 2.0
    assert record["pop_access_15min"] == 30.0
    assert record["existing_firms_access_15min"] == 7.0
    assert record["walk_pop_10min"] == 30.0
    assert record["reachable_cells_15min"] == 2
    assert record["reachable_cells_walk_10min"] == 2


def test_direct_strtree_query_matches_reference_cell_sets() -> None:
    namespace = notebook_namespace()
    destinations = gpd.GeoDataFrame(
        {
            "grid_id": ["g1", "g2", "g3", "g4", "g5"],
            "geometry": [Point(0, 1), Point(1, 1), Point(3, 3), Point(5, 5), Point(9, 9)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    contours = {
        5: box(0, 0, 1, 1),
        10: box(0, 0, 2, 2),
        15: box(0, 0, 2, 2),
        30: box(0, 0, 2, 2).union(box(4, 4, 6, 6)),
    }
    namespace["request_accessibility_isochrones"] = lambda origin, costing, minutes: contours
    position_by_id = {grid_id: position for position, grid_id in enumerate(destinations["grid_id"])}
    actual = namespace["reachable_grid_positions_by_contour"](
        ("g5", 9.0, 9.0), destinations, position_by_id
    )
    for minutes, contour in contours.items():
        candidates = destinations.iloc[
            destinations.sindex.query(box(*contour.bounds), predicate="intersects")
        ]
        expected_ids = {"g5"}
        expected_ids.update(candidates.loc[candidates.geometry.intersects(contour), "grid_id"])
        expected = np.array(sorted(position_by_id[grid_id] for grid_id in expected_ids), dtype=np.int64)
        np.testing.assert_array_equal(actual[minutes], expected)


def test_numpy_aggregation_matches_pandas_reference_for_all_mass_columns() -> None:
    fachgruppe_ids_for_test = [str(1000 + index) for index in range(95)]
    namespace = notebook_namespace(fachgruppe_ids_for_test)
    grid_ids = ["g1", "g2", "g3", "g4"]
    mass_columns = namespace["accessibility_mass_source_columns"]()
    rng = np.random.default_rng(42)
    quarterly_frames = {}
    quarterly_arrays = {}
    for quarter in (1, 2, 3, 4):
        values = rng.random((len(grid_ids), len(mass_columns)), dtype=np.float64)
        frame = pd.DataFrame(values, index=grid_ids, columns=mass_columns)
        quarterly_frames[quarter] = frame
        matrix = np.ascontiguousarray(frame.to_numpy(dtype=np.float64, copy=True))
        matrix.setflags(write=False)
        quarterly_arrays[quarter] = matrix
    reachable_positions = {
        5: np.array([0, 1], dtype=np.int64),
        10: np.array([0, 1, 2, 3], dtype=np.int64),
        15: np.array([0, 1, 2, 3], dtype=np.int64),
        30: np.array([0, 1, 2, 3], dtype=np.int64),
    }
    destinations = gpd.GeoDataFrame(
        {"grid_id": grid_ids, "geometry": [Point(index, 0) for index in range(len(grid_ids))]},
        crs="EPSG:4326",
    )
    contour_shapes = {
        5: box(-1, -1, 1.1, 1),
        10: box(-1, -1, 3.1, 1),
        15: box(-1, -1, 3.1, 1),
        30: box(-1, -1, 4.1, 1),
    }
    namespace["request_accessibility_isochrones"] = lambda origin, costing, minutes: {
        minute: contour_shapes[minute] for minute in minutes
    }
    zero_mass = np.zeros(len(mass_columns), dtype=np.float64)
    zero_mass.setflags(write=False)
    quarter_period_map = {quarter: f"2015Q{quarter}" for quarter in (1, 2, 3, 4)}
    own_columns = namespace["own_cell_mass_output_columns"]()
    contour_columns = {
        minutes: namespace["contour_mass_output_columns"](minutes) for minutes in ACCESS_MINUTES
    }
    records = namespace["accessibility_records_for_origin"](
        ("g2", 0.0, 0.0),
        2015,
        quarterly_arrays,
        quarter_period_map,
        destinations,
        {grid_id: position for position, grid_id in enumerate(grid_ids)},
        destinations,
        {grid_id: position for position, grid_id in enumerate(grid_ids)},
        empty_pt_destinations(),
        zero_mass,
        own_columns,
        namespace["own_cell_walk_mass_output_columns"](),
        contour_columns,
        {minutes: namespace["pedestrian_contour_mass_output_columns"](minutes) for minutes in (5, 10)},
    )
    for record in records:
        frame = quarterly_frames[record["quarter"]]
        for column, expected in zip(own_columns, frame.loc["g2"].to_numpy(), strict=True):
            assert record[column] == pytest.approx(float(expected), abs=1e-9)
        for minutes, positions in reachable_positions.items():
            expected_mass = frame.iloc[positions].sum(axis=0).to_numpy()
            for column, expected in zip(contour_columns[minutes], expected_mass, strict=True):
                assert record[column] == pytest.approx(float(expected), abs=1e-9)

    missing_origin_records = namespace["accessibility_records_for_origin"](
        ("missing", 0.0, 0.0),
        2015,
        quarterly_arrays,
        quarter_period_map,
        destinations,
        {grid_id: position for position, grid_id in enumerate(grid_ids)},
        destinations,
        {grid_id: position for position, grid_id in enumerate(grid_ids)},
        empty_pt_destinations(),
        zero_mass,
        own_columns,
        namespace["own_cell_walk_mass_output_columns"](),
        contour_columns,
        {minutes: namespace["pedestrian_contour_mass_output_columns"](minutes) for minutes in (5, 10)},
    )
    assert all(record[column] == 0.0 for record in missing_origin_records for column in own_columns)


def test_prepared_mass_arrays_are_sorted_aligned_contiguous_and_read_only() -> None:
    namespace = notebook_namespace()
    mass_columns = namespace["accessibility_mass_source_columns"]()
    panel_rows = []
    for quarter in (1, 2, 3, 4):
        for grid_id, base in (("g2", 20.0), ("g1", 10.0)):
            row = {
                "grid_id": grid_id,
                "quarter": quarter,
                **{column: base + offset for offset, column in enumerate(mass_columns)},
            }
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    destinations = gpd.GeoDataFrame(
        {"grid_id": ["g1", "g2"], "geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    namespace["load_yearly_accessibility_panel"] = lambda year: panel.copy()
    namespace["load_accessibility_destinations_for_year"] = lambda frame: destinations.copy()
    namespace["load_all_grid_cells_4326"] = lambda: destinations.copy()
    namespace["load_pt_accessibility_destinations"] = lambda year: empty_pt_destinations()
    prepared = namespace["prepare_accessibility_potential_inputs"](
        2015,
        pd.DataFrame({"grid_id": ["g1"], "lat": [0.0], "lon": [0.0]}),
    )
    assert prepared["destination_position_by_grid_id"] == {"g1": 0, "g2": 1}
    for matrix in prepared["quarterly_mass_arrays"].values():
        assert matrix.dtype == np.float64
        assert matrix.flags.c_contiguous
        assert not matrix.flags.writeable
        assert matrix[0, 0] == 10.0
        assert matrix[1, 0] == 20.0
    assert not prepared["zero_mass_vector"].flags.writeable


def test_documented_skipped_origin_preserves_own_mass_and_routed_nans(tmp_path: Path) -> None:
    namespace = notebook_namespace()
    grid_id = "AT_CRS3035RES100mN2655400E4708200"
    mass_columns = namespace["accessibility_mass_source_columns"]()
    matrix = np.ascontiguousarray(np.arange(1, len(mass_columns) + 1, dtype=np.float64)[None, :])
    matrix.setflags(write=False)
    zero_mass = np.zeros(len(mass_columns), dtype=np.float64)
    zero_mass.setflags(write=False)
    destinations = gpd.GeoDataFrame({"grid_id": [grid_id], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    namespace["request_accessibility_isochrones"] = lambda origin, costing, minutes: (
        pytest.fail("skipped car origin was routed")
        if costing == namespace["POPULATION_ACCESS_COSTING"]
        else {minute: box(-1, -1, 1, 1) for minute in minutes}
    )
    quarters = {quarter: matrix for quarter in (1, 2, 3, 4)}
    periods = {quarter: f"2020Q{quarter}" for quarter in (1, 2, 3, 4)}
    own_columns = namespace["own_cell_mass_output_columns"]()
    contour_columns = {
        minutes: namespace["contour_mass_output_columns"](minutes) for minutes in ACCESS_MINUTES
    }
    records = namespace["accessibility_records_for_origin"](
        (grid_id, 46.8931028030086, 15.081509252261178),
        2020,
        quarters,
        periods,
        destinations,
        {grid_id: 0},
        destinations,
        {grid_id: 0},
        empty_pt_destinations(),
        zero_mass,
        own_columns,
        namespace["own_cell_walk_mass_output_columns"](),
        contour_columns,
        {minutes: namespace["pedestrian_contour_mass_output_columns"](minutes) for minutes in (5, 10)},
    )
    assert len(records) == 4
    assert all(record["own_cell_pop"] == 1.0 for record in records)
    car_routed_columns = [
        column for minutes in ACCESS_MINUTES
        for column in (*namespace["contour_mass_output_columns"](minutes), f"reachable_cells_{minutes}min")
    ]
    assert all(math.isnan(record[column]) for record in records for column in car_routed_columns)
    assert all(not math.isnan(record["walk_pop_10min"]) for record in records)

    part_path = tmp_path / "part_00001.parquet"
    pd.DataFrame(records).to_parquet(part_path, index=False)
    output_path = tmp_path / "accessibility.parquet"
    namespace["write_accessibility_potentials_output"](
        2020,
        [part_path],
        output_path,
        pd.DataFrame({"grid_id": [grid_id]}),
    )
    published = pd.read_parquet(output_path)
    assert published[car_routed_columns].isna().all().all()
    assert published[list(own_columns)].notna().all().all()


def test_old_slice_schema_and_missing_columns_are_rejected(tmp_path: Path) -> None:
    namespace = notebook_namespace()
    old_slice = tmp_path / "old.parquet"
    pd.DataFrame({"grid_id": ["g1"], "year": [2015], "quarter": [1]}).to_parquet(old_slice, index=False)
    assert not namespace["valid_slice_output"](old_slice, {"g1"}, 2015)
    with pytest.raises(KeyError, match="missing expected columns"):
        namespace["write_accessibility_potentials_output"](
            2015, [], tmp_path / "output.parquet", pd.DataFrame({"grid_id": ["g1"]})
        )

    compatible_part = tmp_path / "part_00001.parquet"
    compatible_row = {
        "grid_id": "g1",
        "year": 2015,
        "quarter": 1,
        "period": "2015Q1",
        **{column: 0.0 for column in namespace["accessibility_output_columns"]()},
    }
    pd.DataFrame([compatible_row]).to_parquet(compatible_part, index=False)
    assert namespace["accessibility_part_paths_match_schema"]([compatible_part])


def test_published_nearest_output_is_recognized_for_resume(tmp_path: Path) -> None:
    namespace = notebook_namespace()
    path = tmp_path / "nearest.parquet"
    row = {
        "grid_id": "g1",
        "year": 2015,
    }
    for poi_type in namespace["DESTINATION_TYPES"]:
        row[f"tt_{poi_type}_min"] = 1.0
        row[f"km_{poi_type}"] = 0.1
        row[f"nearest_{poi_type}_id"] = "p1"
        row[f"routing_status_{poi_type}"] = "ok"
    pd.DataFrame([row]).to_parquet(path, index=False)
    assert namespace["valid_nearest_output"](path, {"g1"}, 2015)

    pd.DataFrame([{key: value for key, value in row.items() if key != "routing_status_motorway_exit"}]).to_parquet(path, index=False)
    assert not namespace["valid_nearest_output"](path, {"g1"}, 2015)

    row["routing_status_pt_stop"] = "ok"
    pd.DataFrame([row]).to_parquet(path, index=False)
    assert not namespace["valid_nearest_output"](path, {"g1"}, 2015)

    row.pop("routing_status_pt_stop")
    row["tt_rail_station_min"] = 1.0
    pd.DataFrame([row]).to_parquet(path, index=False)
    assert not namespace["valid_nearest_output"](path, {"g1"}, 2015)


def test_firm_output_keeps_diagonal_and_rejects_zero_same_fachgruppe(tmp_path: Path) -> None:
    namespace = notebook_namespace()
    cell_path = tmp_path / "cells.parquet"
    firm_path = tmp_path / "firms.parquet"
    cells = accessibility_frame(namespace).query("grid_id == 'g1'").copy()
    cells.to_parquet(cell_path, index=False)
    firm_panel = pd.DataFrame({
        "firm_id": ["f1", "f2"],
        "grid_id_100m": ["g1", "g1"],
        "Fachgruppe_ID": pd.Series(["101", "102"], dtype="string"),
        "Fachgruppe_ID_normalized": pd.Series(["101", "102"], dtype="string"),
        "year": [2015, 2015],
        "quarter": [1, 1],
        "period": ["2015Q1", "2015Q1"],
        "included_in_lagged_stock": [True, True],
    })
    namespace["build_firm_quarter_panel_for_year"] = lambda year: firm_panel.copy()
    namespace["output_paths"] = lambda year: {"firm_accessibility": firm_path}
    namespace["generate_firm_accessibility_output"](2015, cell_path)
    result = pd.read_parquet(firm_path)
    assert (result["existing_firms_access_15min"] == 5.0).all()
    assert (result["same_fachgruppe_firms_access_15min"] == 3.0).all()
    assert (result["own_cell_same_fachgruppe_firms"] == 1.0).all()

    for fachgruppe_id in namespace["FACHGRUPPE_IDS"]:
        for minutes in ACCESS_MINUTES:
            cells[f"fachgruppe_{fachgruppe_id}_access_{minutes}min"] = 0.0
    cells.to_parquet(cell_path, index=False)
    with pytest.raises(RuntimeError, match="Same-Fachgruppe accessibility is zero"):
        namespace["generate_firm_accessibility_output"](2015, cell_path)


def test_pedestrian_pt_metrics_deduplicate_routes_across_parent_stations() -> None:
    namespace = notebook_namespace()
    mass_columns = namespace["accessibility_mass_source_columns"]()
    matrix = np.ones((1, len(mass_columns)), dtype=np.float64)
    destinations = gpd.GeoDataFrame({"grid_id": ["g1"], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    pt = gpd.GeoDataFrame({
        "parent_station_id": ["s1", "s2"],
        "pt_departures_weekday": [10.0, 20.0],
        "route_ids": [frozenset({"R1", "R2"}), frozenset({"R2", "R3"})],
        "geometry": [Point(0.1, 0), Point(0.2, 0)],
    }, crs="EPSG:4326")
    calls = []
    def fake_isochrones(origin, costing, minutes):
        calls.append((costing, tuple(minutes)))
        return {minute: box(-1, -1, 1, 1) for minute in minutes}
    namespace["request_accessibility_isochrones"] = fake_isochrones
    record = namespace["accessibility_records_for_origin"](
        ("g1", 0.0, 0.0), 2015, {1: matrix}, {1: "2015Q1"},
        destinations, {"g1": 0}, destinations, {"g1": 0}, pt,
        np.zeros(len(mass_columns)), namespace["own_cell_mass_output_columns"](),
        namespace["own_cell_walk_mass_output_columns"](),
        {minutes: namespace["contour_mass_output_columns"](minutes) for minutes in ACCESS_MINUTES},
        {minutes: namespace["pedestrian_contour_mass_output_columns"](minutes) for minutes in (5, 10)},
    )[0]
    assert record["walk_pt_stops_10min"] == 2
    assert record["walk_pt_departures_10min"] == 30.0
    assert record["walk_pt_routes_10min"] == 3
    assert record["pt_ohne_haltestelle"] == 0
    assert calls == [("auto", (5, 10, 15, 30)), ("pedestrian", (5, 10))]

    no_stop_record = namespace["accessibility_records_for_origin"](
        ("g1", 0.0, 0.0), 2015, {1: matrix}, {1: "2015Q1"},
        destinations, {"g1": 0}, destinations, {"g1": 0}, empty_pt_destinations(),
        np.zeros(len(mass_columns)), namespace["own_cell_mass_output_columns"](),
        namespace["own_cell_walk_mass_output_columns"](),
        {minutes: namespace["contour_mass_output_columns"](minutes) for minutes in ACCESS_MINUTES},
        {minutes: namespace["pedestrian_contour_mass_output_columns"](minutes) for minutes in (5, 10)},
    )[0]
    assert no_stop_record["walk_pt_stops_10min"] == 0
    assert no_stop_record["walk_pt_departures_10min"] == 0.0
    assert no_stop_record["walk_pt_routes_10min"] == 0
    assert no_stop_record["pt_ohne_haltestelle"] == 1


def test_partitioned_fachgruppe_output_and_duckdb_validation(tmp_path: Path) -> None:
    namespace = notebook_namespace()
    wide_path = tmp_path / "accessibility.parquet"
    pedestrian_path = tmp_path / "pedestrian.parquet"
    long_path = tmp_path / "fachgruppe.parquet"
    accessibility_frame(namespace).to_parquet(wide_path, index=False)
    namespace["split_fachgruppe_accessibility"](wide_path, pedestrian_path, long_path)
    pedestrian = pd.read_parquet(pedestrian_path)
    assert {"walk_pop_5min", "walk_pt_routes_10min", "pt_ohne_haltestelle"}.issubset(pedestrian.columns)

    files = sorted(long_path.rglob("*.parquet"))
    assert len(files) == len(namespace["FACHGRUPPE_IDS"])
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?, hive_partitioning = true)",
            [(long_path / "**" / "*.parquet").as_posix()],
        ).fetchone()[0]
    assert rows == 2 * 4 * len(namespace["FACHGRUPPE_IDS"])

    nearest_path = tmp_path / "nearest.parquet"
    firm_path = tmp_path / "firm.parquet"
    pd.DataFrame({"grid_id": ["g1", "g2"]}).to_parquet(nearest_path, index=False)
    pd.DataFrame({"year": [2015]}).to_parquet(firm_path, index=False)
    namespace["canonical_output_paths"] = lambda year: {
        "nearest": nearest_path,
        "potentials": wide_path,
        "pedestrian_accessibility": pedestrian_path,
        "fachgruppe_accessibility": long_path,
        "firm_accessibility": firm_path,
    }
    validated = namespace["validate_published_year"](2015, pd.DataFrame({"grid_id": ["g1", "g2"]}))
    assert validated["fachgruppe_rows"] == rows
    assert validated["fachgruppe_row_groups"] <= len(namespace["FACHGRUPPE_IDS"])


def test_partitioned_writer_coalesces_batches_into_128k_row_groups(tmp_path: Path) -> None:
    namespace = notebook_namespace(["101"])
    row_count = 128_001
    wide_path = tmp_path / "accessibility.parquet"
    pedestrian_path = tmp_path / "pedestrian.parquet"
    long_path = tmp_path / "fachgruppe.parquet"
    frame = pd.DataFrame({
        "grid_id": [f"g{index:06d}" for index in range(row_count)],
        "year": np.full(row_count, 2015, dtype=np.int16),
        "quarter": np.ones(row_count, dtype=np.int8),
        "period": np.full(row_count, "2015Q1", dtype=object),
        "own_cell_pop": np.ones(row_count),
        "own_cell_firms": np.ones(row_count),
        "own_cell_walk_pop": np.ones(row_count),
        "own_cell_walk_firms": np.ones(row_count),
        "own_cell_fachgruppe_101_firms": np.ones(row_count),
        "own_cell_walk_fachgruppe_101_firms": np.ones(row_count),
        "pt_ohne_haltestelle": np.zeros(row_count, dtype=np.int8),
        "created_at": np.full(row_count, "test", dtype=object),
    })
    for minutes in ACCESS_MINUTES:
        frame[f"pop_access_{minutes}min"] = 1.0
        frame[f"existing_firms_access_{minutes}min"] = 1.0
        frame[f"fachgruppe_101_access_{minutes}min"] = 1.0
        frame[f"reachable_cells_{minutes}min"] = 1
    for minutes in (5, 10):
        frame[f"walk_pop_{minutes}min"] = 1.0
        frame[f"walk_firms_{minutes}min"] = 1.0
        frame[f"walk_fachgruppe_101_firms_{minutes}min"] = 1.0
        frame[f"walk_pt_stops_{minutes}min"] = 1
        frame[f"walk_pt_departures_{minutes}min"] = 1.0
        frame[f"walk_pt_routes_{minutes}min"] = 1
        frame[f"reachable_cells_walk_{minutes}min"] = 1
    frame.to_parquet(wide_path, index=False, row_group_size=50_000)
    namespace["split_fachgruppe_accessibility"](wide_path, pedestrian_path, long_path)

    files = list(long_path.rglob("*.parquet"))
    assert len(files) == 1
    metadata = pq.ParquetFile(files[0]).metadata
    assert metadata.num_row_groups == 2
    assert [metadata.row_group(index).num_rows for index in range(2)] == [128_000, 1]
