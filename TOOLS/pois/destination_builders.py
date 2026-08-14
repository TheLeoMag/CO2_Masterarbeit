"""Year-specific public-transport and motorway-ramp routing destinations."""

from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely import get_point


CRS_ANALYSIS = "EPSG:3035"
CRS_ROUTING = "EPSG:4326"
OBSERVED_PT_YEARS = (2016, 2017, 2018, 2019, 2020, 2022, 2025)
PT_YEAR_SOURCES = {
    2015: (2016,), 2016: (2016,), 2017: (2017,), 2018: (2018,),
    2019: (2019,), 2020: (2020,), 2021: (2020, 2022), 2022: (2022,),
    2023: (2022,), 2024: (2025,), 2025: (2025,),
}

# User-supplied train-station name rule. Keep this exact string in exported metadata.
TRAIN_STATION_NAME_REGEX = r"(?ix)^(?!.*\b(?:BH|WP|AO)\b)(?!.*(?:Busbahnhof|Fernbusbahnhof|Güterbahnhof|Verschiebebahnhof))(?!.*\bAbzw\.?\s+.*Bahnhof\b)(?!.*\bBahnhof(?:straße|str\.?|weg|gürtel)\b)(?!.*\b(?:ehemaliger|ehem\.?|Alter)\s+Bahnhof\b)(?!.*\b(?:Seilbahn|Gondelbahn|Gondlbahn|Sesselbahn|Schloßbergbahn|Autobahn)\b)(?!.*\b(?:Bahndurchlass|Bahnwärterhaus|Bahndamm|Bahnweg)\b)(?!.*(?:\[ALT\]|\(ALT\)|ersetzt\s+durch|\bPLAN\b))(?=.*(?:Bahnhof\b|Bahnhaltestelle\b|\bHbf\b|\bBf\b|\bS[- ]?Bahn\b)).+$"
TRAIN_STATION_CLUSTER_M = 1_000
RAMP_ENDPOINT_SNAP_M = 10
RAMP_CLUSTER_M = 250


def _empty_geodataframe() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=CRS_ROUTING), crs=CRS_ROUTING)


def _transport_path(project_dir: Path, year: int) -> Path:
    return project_dir / "OGD" / "Public_Transport" / f"public_transport_weekday_stop_frequency_{year}.geoparquet"


def _read_observed_transport(project_dir: Path, year: int) -> gpd.GeoDataFrame:
    path = _transport_path(project_dir, year)
    if not path.exists():
        raise FileNotFoundError(f"Missing public-transport GeoParquet: {path}")
    stops = gpd.read_parquet(path).to_crs(CRS_ROUTING)
    required = {"station_id", "station_name", "weekday_school_departures", "weekday_holiday_departures", "geometry"}
    missing = sorted(required - set(stops.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    stops = stops.dropna(subset=["station_id", "geometry"]).copy()
    stops["station_id"] = pd.to_numeric(stops["station_id"], errors="raise").astype("int64")
    return stops


def _join_values(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna() if str(value)}))


def yearly_transport_stops(project_dir: Path, year: int) -> gpd.GeoDataFrame:
    """Return the assigned stop layer for a routing year with explicit provenance."""
    if year not in PT_YEAR_SOURCES:
        raise ValueError(f"No public-transport source assignment for {year}")
    source_years = PT_YEAR_SOURCES[year]
    frames = []
    for source_year in source_years:
        frame = _read_observed_transport(project_dir, source_year).copy()
        frame["pt_source_year"] = source_year
        frame["source_stop_id"] = frame["station_id"].astype(str)
        frames.append(frame)

    if len(frames) == 1:
        stops = frames[0].copy()
        stops["source_years"] = str(source_years[0])
        stops["source_stop_ids"] = stops["source_stop_id"]
        stops["imputation_method"] = "observed" if year == source_years[0] else f"nearest_observed_year_{source_years[0]}"
    else:
        combined = pd.concat(frames, ignore_index=True)
        rows = []
        for station_id, group in combined.groupby("station_id", sort=True):
            group = group.sort_values("pt_source_year")
            row = group.iloc[0].copy()
            row["station_id"] = station_id
            row["weekday_school_departures"] = pd.to_numeric(group["weekday_school_departures"], errors="coerce").mean()
            row["weekday_holiday_departures"] = pd.to_numeric(group["weekday_holiday_departures"], errors="coerce").mean()
            row["weekday_avg_departures"] = (row["weekday_school_departures"] + row["weekday_holiday_departures"]) / 2
            row["source_years"] = _join_values(group["pt_source_year"])
            row["source_stop_ids"] = _join_values(group.apply(lambda r: f"{r.pt_source_year}:{r.station_id}", axis=1))
            row["imputation_method"] = "mean_2020_2022_matching_station_id" if len(group) == 2 else "2021_one_sided_stop_retained"
            rows.append(row)
        stops = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_ROUTING)

    stops["year"] = int(year)
    stops["weekday_school_departures"] = pd.to_numeric(stops["weekday_school_departures"], errors="coerce").fillna(0.0)
    stops["weekday_holiday_departures"] = pd.to_numeric(stops["weekday_holiday_departures"], errors="coerce").fillna(0.0)
    stops["weekday_avg_departures"] = (stops["weekday_school_departures"] + stops["weekday_holiday_departures"]) / 2
    return stops.to_crs(CRS_ROUTING)


def _components_within_distance(frame: gpd.GeoDataFrame, distance_m: float) -> list[list[int]]:
    if frame.empty:
        return []
    projected = frame.to_crs(CRS_ANALYSIS).reset_index(drop=True)
    parents = list(range(len(projected)))

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for left, right in zip(*projected.sindex.query(projected.geometry, predicate="dwithin", distance=distance_m)):
        left_root, right_root = find(int(left)), find(int(right))
        if left_root != right_root:
            parents[right_root] = left_root
    components: dict[int, list[int]] = {}
    for index in range(len(projected)):
        components.setdefault(find(index), []).append(index)
    return list(components.values())


def rail_station_candidates(stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    names = stops["station_name"].fillna("").astype(str)
    selected = stops[names.str.contains(re.compile(TRAIN_STATION_NAME_REGEX), na=False)].copy()
    selected["train_station_name_regex"] = TRAIN_STATION_NAME_REGEX
    selected["train_station_candidate"] = True
    return selected


def rail_station_destinations(stops: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    candidates = rail_station_candidates(stops)
    rows = []
    for component_number, member_positions in enumerate(_components_within_distance(candidates, TRAIN_STATION_CLUSTER_M), start=1):
        members = candidates.iloc[member_positions].copy()
        representative = members.sort_values(["weekday_avg_departures", "station_id"], ascending=[False, True]).iloc[0]
        member_ids = sorted(members["station_id"].astype(int).tolist())
        rows.append({
            "name": representative["station_name"],
            "ref": str(int(representative["station_id"])),
            "year": int(representative["year"]),
            "poi_type": "rail_station",
            "source_poi_id": f"rail_cluster_{representative['year']}_{component_number:04d}",
            "source_file": "public_transport_weekday_stop_frequency.geoparquet",
            "source_schema": "public_transport_train_station_name_rule",
            "source_note": "public_transport_derived_train_station",
            "source_years": _join_values(members["source_years"]),
            "source_stop_ids": _join_values(members["source_stop_ids"]),
            "imputation_method": _join_values(members["imputation_method"]),
            "provenance": "public_transport_train_station_cluster",
            "train_station_name_regex": TRAIN_STATION_NAME_REGEX,
            "station_cluster_id": f"rail_station_{representative['year']}_{component_number:04d}",
            "station_cluster_member_count": len(members),
            "station_cluster_member_ids": "|".join(map(str, member_ids)),
            "station_cluster_representative_id": int(representative["station_id"]),
            "pt_departures_weekday": float(representative["weekday_avg_departures"]),
            "static_destination": True,
            "geometry": representative.geometry,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_ROUTING), candidates


def pt_stop_destinations(stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = stops.copy()
    output["name"] = output["station_name"].astype("string")
    output["ref"] = output["station_id"].astype(str)
    output["poi_type"] = "pt_stop"
    output["source_poi_id"] = output["station_id"].astype(str)
    output["source_file"] = "public_transport_weekday_stop_frequency.geoparquet"
    output["source_schema"] = "public_transport_weekday_stop_frequency"
    output["source_note"] = "public_transport_stop"
    output["provenance"] = "public_transport_assigned_year"
    output["pt_departures_weekday"] = output["weekday_avg_departures"].astype(float)
    output["static_destination"] = True
    return output


def _ramp_components(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    links = roads[roads["highway"].eq("motorway_link")].reset_index(drop=True)
    if links.empty:
        return gpd.GeoDataFrame(columns=["off_ramp", "on_ramp", "geometry"], geometry="geometry", crs=CRS_ANALYSIS)
    endpoints = gpd.GeoDataFrame(
        {"link": list(range(len(links))) * 2, "end": [False] * len(links) + [True] * len(links)},
        geometry=pd.concat([gpd.GeoSeries(get_point(links.geometry.values, 0), crs=CRS_ANALYSIS), gpd.GeoSeries(get_point(links.geometry.values, -1), crs=CRS_ANALYSIS)], ignore_index=True),
        crs=CRS_ANALYSIS,
    )
    components = _components_within_distance(endpoints, RAMP_ENDPOINT_SNAP_M)
    endpoint_to_node = {endpoint: node for node, members in enumerate(components) for endpoint in members}
    endpoints["node"] = [endpoint_to_node[index] for index in range(len(endpoints))]
    link_nodes = endpoints.pivot(index="link", columns="end", values="node")
    nodes = gpd.GeoDataFrame(geometry=endpoints.dissolve(by="node").geometry.centroid, crs=CRS_ANALYSIS)
    other_roads = roads[roads["highway"].ne("motorway_link")]
    pairs = other_roads.sindex.query(nodes.geometry, predicate="dwithin", distance=RAMP_ENDPOINT_SNAP_M)
    connected_nodes = nodes.index.to_numpy()[pairs[0]]
    connected_types = other_roads.iloc[pairs[1]]["highway"].to_numpy()
    motorway_nodes = set(connected_nodes[connected_types == "motorway"])
    local_nodes = set(connected_nodes[connected_types != "motorway"])
    directed = {node: set() for node in nodes.index}
    undirected = {node: set() for node in nodes.index}
    for link, oneway in enumerate(links["other_tags"].fillna("").str.contains('"oneway"=>"yes"', regex=False)):
        start, end = link_nodes.loc[link, False], link_nodes.loc[link, True]
        directed[start].add(end)
        if not oneway:
            directed[end].add(start)
        undirected[start].add(end)
        undirected[end].add(start)

    def reachable(starts: set, targets: set) -> bool:
        seen, queue = set(starts), list(starts)
        while queue:
            node = queue.pop()
            if node in targets:
                return True
            for neighbour in directed[node] - seen:
                seen.add(neighbour)
                queue.append(neighbour)
        return False

    visited, rows = set(), []
    for seed in nodes.index:
        if seed in visited:
            continue
        component, queue = {seed}, [seed]
        while queue:
            node = queue.pop()
            for neighbour in undirected[node] - component:
                component.add(neighbour)
                queue.append(neighbour)
        visited.update(component)
        motorway_contacts, local_contacts = component & motorway_nodes, component & local_nodes
        if motorway_contacts and local_contacts:
            off_ramp = reachable(motorway_contacts, local_contacts)
            on_ramp = reachable(local_contacts, motorway_contacts)
            if off_ramp or on_ramp:
                rows.append({"off_ramp": off_ramp, "on_ramp": on_ramp, "geometry": nodes.loc[list(component)].geometry.union_all().centroid})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_ANALYSIS)


def gip_motorway_exit_destinations(year: int, gip_pbf_path: Path, mask_wgs84) -> gpd.GeoDataFrame:
    roads = pyogrio.read_dataframe(gip_pbf_path, layer="lines", columns=["highway", "other_tags"], mask=mask_wgs84).to_crs(CRS_ANALYSIS)
    components = _ramp_components(roads)
    rows = []
    for cluster_number, member_positions in enumerate(_components_within_distance(components, RAMP_CLUSTER_M), start=1):
        members = components.iloc[member_positions]
        centroid = members.geometry.buffer(RAMP_CLUSTER_M / 2).union_all().centroid
        rows.append({
            "name": f"GIP motorway interchange {cluster_number}", "ref": str(cluster_number), "year": year,
            "poi_type": "motorway_exit", "source_poi_id": f"gip_ramp_{year}_{cluster_number:04d}",
            "source_file": gip_pbf_path.name, "source_schema": "gip_motorway_link_interchange_components",
            "source_note": "gip_derived_motorway_ramp", "source_years": str(year), "source_stop_ids": pd.NA,
            "imputation_method": "observed", "provenance": "gip_motorway_ramp_cluster",
            "ramp_cluster_id": f"motorway_exit_{year}_{cluster_number:04d}",
            "ramp_component_count": len(members), "off_ramp": bool(members["off_ramp"].any()),
            "on_ramp": bool(members["on_ramp"].any()), "static_destination": False, "geometry": centroid,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_ANALYSIS).to_crs(CRS_ROUTING)
