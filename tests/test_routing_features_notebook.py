from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gc
import itertools
import json
import math
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from shapely.geometry import Point, box, shape

from ANAL.routing.routing_utils import (
    ACCESS_MINUTES,
    fachgruppe_access_columns,
    fachgruppe_ids,
    fachgruppe_stock_columns,
    main_access_columns,
)
from ANAL.routing.destination_builders import pt_stop_destinations, yearly_transport_stops


NOTEBOOK_PATH = Path(__file__).parents[1] / "ANAL" / "routing" / "06_generate_features.ipynb"


def notebook_namespace(fachgruppe_ids_for_test: list[str] | None = None) -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = dict(globals())
    namespace.update({
        "FACHGRUPPE_IDS": fachgruppe_ids_for_test or ["101", "102"],
        "START_YEAR": 2015,
        "END_YEAR": 2025,
        "CELL_SIZE": 100,
        "RESOLUTION": "100m",
        "ROUTING_DATA": Path("."),
        "FEATURE_ROOT": Path("."),
        "POPULATION_ACCESS_CONTOURS_MIN": list(ACCESS_MINUTES),
        "PEDESTRIAN_ACCESS_CONTOURS_MIN": [5, 10],
        "NEAREST_INFRA_DESTINATION_STEP_SHARE": 0.10,
        "DESTINATION_CHUNK_SIZE": 100,
    })
    cells = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]}
    for cell_id in ("progress-display", "48ba6cda", "433c477a", "8b608cd1"):
        exec(compile(cells[cell_id], f"notebook-cell-{cell_id}", "exec"), namespace)
    orchestration = cells["30e73ef1"]
    exec(compile(orchestration.split("if RUN_MODE not in")[0], "notebook-orchestration-definitions", "exec"), namespace)
    return namespace


def progress_namespace(kernel=True):
    """Capture display protocol calls without requiring a browser or sleeping."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "progress-display")
    namespace = dict(math=math, threading=threading, time=time)
    exec(compile(source, "progress-display", "exec"), namespace)
    clock, displays, updates = [0.0], [], []
    namespace["time"] = SimpleNamespace(monotonic=lambda: clock[0])
    namespace["get_ipython"] = lambda: SimpleNamespace(kernel=object()) if kernel else None
    handle = SimpleNamespace(update=lambda bundle, **kwargs: updates.append(bundle))
    def show(bundle, **kwargs):
        assert kwargs == {"raw": True, "display_id": True}
        displays.append(bundle)
        return handle
    namespace["display"] = show
    return namespace, clock, displays, updates, handle


def test_progress_updates_one_display_and_throttles_origin_events():
    namespace, clock, displays, updates, _ = progress_namespace()
    progress = namespace["NotebookProgress"](2000, "Routing <example>")
    for _ in range(1000):
        progress.update()
    assert len(displays) == 1 and not updates
    assert "&lt;example&gt;" in displays[0]["text/html"]
    clock[0] = 2.1
    progress.update(detail="12 matrix requests")
    assert len(updates) == 1
    assert "1,001/2,000" in updates[0]["text/plain"]
    assert "12 matrix requests" in updates[0]["text/plain"]
    progress.close()
    progress.close()
    assert len(displays) == 1 and len(updates) == 2
    assert "Stopped" in updates[-1]["text/plain"]


def test_progress_resume_eta_uses_only_newly_processed_origins():
    namespace, clock, _, updates, _ = progress_namespace()
    progress = namespace["NotebookProgress"](100, "Resume", initial=90)
    clock[0] = 2.0
    progress.update()
    assert "ETA 00:00:18" in updates[-1]["text/plain"]
    progress.update(9)
    progress.close()
    assert "100/100" in updates[-1]["text/plain"]
    assert "Complete" in updates[-1]["text/plain"]


def test_progress_concurrent_updates_preserve_counts():
    namespace, _, displays, updates, _ = progress_namespace()
    progress = namespace["NotebookProgress"](1000, "Parallel routing")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: progress.update(), range(1000)))
    progress.close()
    assert progress.n == 1000
    assert len(displays) == 1 and len(updates) == 1
    assert "100.0%" in updates[-1]["text/plain"]


def test_progress_falls_back_to_coarse_logs_if_display_fails(capsys):
    namespace, clock, displays, _, handle = progress_namespace()
    progress = namespace["NotebookProgress"](1000, "Routing")
    def broken_display(*args, **kwargs):
        raise RuntimeError("Display transport unavailable")
    handle.update = broken_display
    clock[0] = 2.0
    progress.update()
    assert not progress.rich
    for _ in range(100):
        progress.update()
    assert len(capsys.readouterr().out.splitlines()) == 1
    clock[0] = 62.0
    progress.update()
    progress.close()
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert len(displays) == 1 and progress.n == 102


def test_progress_without_a_notebook_does_not_print_per_origin(capsys):
    namespace, clock, displays, updates, _ = progress_namespace(kernel=False)
    progress = namespace["NotebookProgress"](1000, "Terminal")
    for _ in range(1000):
        progress.update()
    progress.close()
    assert not displays and not updates
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.parametrize("threaded", [False, True])
def test_progress_real_kernel_keeps_one_saved_output(threaded):
    import nbformat
    from nbclient import NotebookClient
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "progress-display")
    work = "progress = NotebookProgress(2000, 'Kernel test')\nfor _ in range(2000): progress.update()\nprogress.close()\n"
    if threaded:
        work = ("from concurrent.futures import ThreadPoolExecutor\ndef stage():\n"
                + "\n".join("    "+line for line in work.splitlines())
                + "\nwith ThreadPoolExecutor(max_workers=1) as pool: pool.submit(stage).result()\n")
    source = "import math, threading, time\n"+source+"\n"+work
    fixture = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    executed = NotebookClient(fixture, timeout=60, kernel_name="python3").execute()
    outputs = executed.cells[0].outputs
    assert len(outputs) == 1 and outputs[0].output_type == "display_data"
    assert "2,000/2,000" in outputs[0].data["text/plain"]
    assert "Complete" in outputs[0].data["text/plain"]
    assert "<progress" in outputs[0].data["text/html"]


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


def test_yearly_transport_stops_reads_prepared_target_year(tmp_path: Path) -> None:
    output_dir = tmp_path / "OGD" / "Public_Transport"
    output_dir.mkdir(parents=True)
    frame = gpd.GeoDataFrame({
        "year": [2016], "frequency_source_year": [2017],
        "imputation_method": ["full_2017_proxy"], "station_id": [1],
        "station_name": ["Test"], "weekday_school_departures": [10.0],
        "weekday_holiday_departures": [8.0], "weekday_route_ids": ["R1"],
    }, geometry=[Point(15.4, 47.1)], crs="EPSG:4326")
    frame.to_parquet(output_dir / "public_transport_weekday_stop_frequency_2016.geoparquet")

    result = yearly_transport_stops(tmp_path, 2016)

    assert result["year"].tolist() == [2016]
    assert result["pt_source_year"].tolist() == [2017]
    assert result["imputation_method"].tolist() == ["full_2017_proxy"]


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
        "grid_id": ["g1", "g1"],
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


@pytest.mark.parametrize("source_ids,target_ids", [([0], [0, 1]), ([0, 1], [0]), ([0, 1, 2], [0, 1, 2]), ([0, 2], [4, 6])])
def test_disconnected_matrix_batches_preserve_valid_pairs_and_order(tmp_path, monkeypatch, source_ids, target_ids):
    monkeypatch.chdir(tmp_path)
    namespace = notebook_namespace()
    namespace["valhalla_endpoint"] = lambda: "http://mock"
    calls = []
    def post(url, json, timeout):
        sources = [int(row["lon"]) for row in json["sources"]]
        targets = [int(row["lon"]) for row in json["targets"]]
        calls.append((sources, targets))
        if len({i % 2 for i in sources + targets}) != 1:
            return SimpleNamespace(ok=False, status_code=400, text="Locations are in unconnected regions.")
        matrix = [[{"time": 100 * i + j + 1, "distance": 1.} for j in targets] for i in sources]
        return SimpleNamespace(ok=True, json=lambda: {"sources_to_targets": matrix})
    namespace["requests"] = SimpleNamespace(post=post, RequestException=requests.RequestException)
    sources = pd.DataFrame({"lat": 47., "lon": source_ids})
    targets = pd.DataFrame({"lat": 47., "lon": target_ids})
    actual = namespace["route_matrix"](sources, targets)
    expected = [[{"time": 100 * i + j + 1, "distance": 1.} if i % 2 == j % 2 else {"status": 1}
                 for j in target_ids] for i in source_ids]
    assert actual == expected
    if len({i % 2 for i in source_ids + target_ids}) == 1:
        assert len(calls) == 1


@pytest.mark.parametrize("matrix", [[], [[None]], [[{}, {}]]])
def test_matrix_rejects_truncated_or_malformed_results(tmp_path, monkeypatch, matrix):
    monkeypatch.chdir(tmp_path)
    namespace = notebook_namespace()
    namespace["valhalla_endpoint"] = lambda: "http://mock"
    namespace["requests"] = SimpleNamespace(post=lambda *a, **k: SimpleNamespace(
        ok=True, json=lambda: {"sources_to_targets": matrix}), RequestException=requests.RequestException)
    frame = pd.DataFrame({"lat": [47.], "lon": [15.]})
    with pytest.raises(RuntimeError, match="incomplete or malformed"):
        namespace["route_matrix"](frame, frame)


@pytest.mark.parametrize("column,value", [("founding_date", "not-a-date"), ("exit_date", "NaT"), ("exit_date", "2020-13-10"), ("exit_date", 44205)])
def test_routing_rejects_corrupt_normalized_dates(tmp_path, monkeypatch, column, value):
    monkeypatch.chdir(tmp_path)
    namespace=notebook_namespace()
    frame=pd.DataFrame({"firm_id":[1],"grid_id":["g1"],"Fachgruppe_ID":["101"],
        "founding_date":["2020-05-04"],"exit_date":["2021-01-01"]})
    frame[column]=[value]
    path=tmp_path/'firms.parquet'
    frame.to_parquet(path,index=False)
    namespace['FIRMS_PATH']=path
    with pytest.raises(ValueError,match='Invalid normalized'):
        namespace['load_firm_accessibility_source']()


def validation_inputs(tmp_path, monkeypatch):
    """Use the actual 06 writers, including partitioned long output, as 07 inputs."""
    import hashlib
    monkeypatch.chdir(tmp_path)
    namespace = notebook_namespace()
    data = tmp_path / "data"
    features = data / "routing/features/2015"
    features.mkdir(parents=True)
    panel_path = data / "panel.parquet"
    wide = accessibility_frame(namespace)
    pd.DataFrame({"grid_id": ["g1", "g2"]}).to_parquet(panel_path, index=False)
    active_path = data / "active.parquet"
    pd.DataFrame({"grid_id": ["g1", "g2"]}).to_parquet(active_path, index=False)
    firms_path = data / "firms_assigned_100m.geoparquet"
    pd.DataFrame({"firm_id": [1, 2], "grid_id": ["g1", "g2"], "Fachgruppe_ID": ["101", "102"],
        "founding_date": pd.to_datetime(["2014-01-01", "2015-12-31 12:00"], format="mixed"),
        "exit_date": pd.to_datetime(["2015-06-30 12:00", None]), "exit_observed": [True, False]}).to_parquet(firms_path, index=False)
    namespace["FIRMS_PATH"] = firms_path
    main = features / "accessibility_potentials_100m.parquet"
    walk = features / "pedestrian_accessibility_quarter_100m.parquet"
    long = features / "fachgruppe_accessibility_quarter_100m.parquet"
    firm = features / "firm_accessibility_quarter_100m.parquet"
    namespace["output_paths"] = lambda _: {"firm_accessibility": firm}
    wide.to_parquet(main, index=False)
    namespace["generate_firm_accessibility_output"](2015, main)
    # These boundaries used to omit firm 2 and add a spurious Q3 to firm 1.
    output = pd.read_parquet(firm)
    assert list(zip(output.firm_id, output.quarter)) == [(1, 1), (1, 2), (2, 4)]
    assert output.included_in_lagged_stock.tolist() == [True, True, False]
    namespace["split_fachgruppe_accessibility"](main, walk, long)
    types = namespace["DESTINATION_TYPES"]
    nearest = pd.DataFrame({"grid_id": ["g1", "g2"], "year": [2015, 2015]})
    for kind in types:
        nearest[f"tt_{kind}_min"] = [2., 3.]
        nearest[f"km_{kind}"] = [1., 2.]
        nearest[f"nearest_{kind}_id"] = kind + "_target"
        nearest[f"routing_status_{kind}"] = "ok"
    nearest_path = features / "nearest_infrastructure_100m.parquet"
    nearest.to_parquet(nearest_path, index=False)
    poi_dir = data / "pois"
    poi_dir.mkdir()
    pd.DataFrame({"poi_id": [t + "_target" for t in types], "poi_type": types, "year": 2015}).to_parquet(poi_dir / "austria-2015-pois.geoparquet", index=False)
    status_path = data / "status.csv"
    pd.DataFrame({"year": [2015], "status": ["done"], "firm_mass_source": ["active_firms_tminus1"]}).to_csv(status_path, index=False)
    skip_path = data / "skips.csv"
    pd.DataFrame(columns=["year", "grid_id", "lat", "lon", "reason"]).to_csv(skip_path, index=False)
    pbf_dir = data / "pbf"
    pbf_dir.mkdir()
    (pbf_dir / "2015.osm.pbf").write_bytes(b"software fixture")
    graph_path = data / "graphs.csv"
    pd.DataFrame({"year": [2015], "status": ["done"], "manifest_path_wsl": ["fixture"],
        "valhalla_image": ["fixture"], "valhalla_image_id": ["fixture"],
        "gip_snapshot_sha256": [hashlib.sha256(b"software fixture").hexdigest()]}).to_csv(graph_path, index=False)
    scope = dict(Path=Path, hashlib=hashlib, json=json, duckdb=duckdb, np=np, pd=pd,
        ACCESS_MINUTES=ACCESS_MINUTES, main_access_columns=main_access_columns,
        PANEL_PATH=panel_path, ACTIVE_CELLS_PATH=active_path, ANAL_DATA=data,
        ROUTING_DATA=data / "routing", STATUS_PATH=status_path, SKIPPED_ACCESSIBILITY_PATH=skip_path,
        GRAPH_STATUS_PATH=graph_path, POI_DIR=poi_dir, PBF_DIR=pbf_dir, REPORT_PATH=data / "report.csv",
        WORK_ROOT=data / "routing/work/routing_features", YEARS=[2015], RESOLUTION="100m",
        FACHGRUPPE_IDS=["101", "102"], GRAPH_VALHALLA_IMAGE="fixture", GRAPH_VALHALLA_IMAGE_ID="fixture")
    return scope, dict(nearest=nearest_path, main=main, walk=walk, long=long, firm=firm)


def execute_validation(scope):
    path = NOTEBOOK_PATH.with_name("07_validate_outputs.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "8dfe46ec"))
    exec(compile(source, "actual-notebook-07", "exec"), scope)


def test_07_accepts_actual_writer_outputs_without_pandas_parquet_loads(tmp_path, monkeypatch):
    scope, _ = validation_inputs(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("07 must not materialize the large products in Pandas")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    execute_validation(scope)
    assert scope["validation"].status.eq("OK").all()


@pytest.mark.parametrize("defect", ["nearest_grid", "nearest_year", "nearest_nan", "main_grid", "main_period",
    "infinity", "missing_schema", "missing_long_quarter", "duplicate_long_key", "unmatched_firm", "missing_firm_quarter", "wrong_long_mass"])
def test_07_rejects_incomplete_or_inconsistent_products(tmp_path, monkeypatch, defect):
    scope, paths = validation_inputs(tmp_path, monkeypatch)
    if defect in {"missing_long_quarter", "duplicate_long_key", "wrong_long_mass"}:
        for i, path in enumerate(sorted(paths["long"].rglob("*.parquet"))):
            frame = pq.ParquetFile(path).read().to_pandas()
            if defect == "missing_long_quarter": frame = frame.loc[frame.quarter.ne(4)]
            if defect == "duplicate_long_key" and i == 0: frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
            if defect == "wrong_long_mass" and i == 0: frame.loc[0, "same_fachgruppe_firms_access_30min"] = 999.
            frame.to_parquet(path, index=False)
    else:
        product = "nearest" if defect.startswith("nearest") else "firm" if defect in {"unmatched_firm", "missing_firm_quarter"} else "main"
        path = paths[product]
        frame = pd.read_parquet(path)
        if defect in {"nearest_grid", "main_grid", "unmatched_firm"}: frame.loc[0, "grid_id"] = "unmatched_grid"
        elif defect == "nearest_year": frame.loc[0, "year"] = 1900
        elif defect == "nearest_nan": frame.loc[0, "tt_motorway_exit_min"] = np.nan
        elif defect == "main_period": frame.loc[0, "period"] = "wrong"
        elif defect == "infinity": frame.loc[0, "pop_access_15min"] = np.inf
        elif defect == "missing_schema": frame = frame.drop(columns="own_cell_pop")
        elif defect == "missing_firm_quarter": frame = frame.iloc[1:]
        frame.to_parquet(path, index=False)
    with pytest.raises(AssertionError, match="Routing validation failed"):
        execute_validation(scope)
    assert scope["validation"].status.eq("CHECK").any()


def test_07_allows_only_documented_missing_isochrone_masses(tmp_path, monkeypatch):
    scope, paths = validation_inputs(tmp_path, monkeypatch)
    pd.DataFrame({"year": [2015], "grid_id": ["g1"], "lat": [47.], "lon": [15.], "reason": ["fixture skip"]}).to_csv(scope["SKIPPED_ACCESSIBILITY_PATH"], index=False)
    for name in ["main", "walk", "firm"]:
        frame = pd.read_parquet(paths[name])
        for column in frame:
            if "min" in column and pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = frame[column].astype(float)
                frame.loc[frame.grid_id.eq("g1"), column] = np.nan
        frame.to_parquet(paths[name], index=False)
    for path in paths["long"].rglob("*.parquet"):
        frame = pq.ParquetFile(path).read().to_pandas()
        for column in frame:
            if "min" in column: frame.loc[frame.grid_id.eq("g1"), column] = np.nan
        frame.to_parquet(path, index=False)
    execute_validation(scope)
    assert scope["validation"].status.eq("OK").all()
