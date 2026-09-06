"""Year-specific public-transport and motorway-ramp routing destinations."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely import get_point


CRS_ANALYSIS = "EPSG:3035"
CRS_ROUTING = "EPSG:4326"
RAMP_ENDPOINT_SNAP_M = 10
RAMP_CLUSTER_M = 250


def _empty_geodataframe() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=CRS_ROUTING), crs=CRS_ROUTING)


def _transport_path(project_dir: Path, year: int) -> Path:
    return project_dir / "OGD" / "Public_Transport" / f"public_transport_weekday_stop_frequency_{year}.geoparquet"


def _read_transport(project_dir: Path, year: int) -> gpd.GeoDataFrame:
    path = _transport_path(project_dir, year)
    if not path.exists():
        raise FileNotFoundError(f"Missing public-transport GeoParquet: {path}")
    stops = gpd.read_parquet(path).to_crs(CRS_ROUTING)
    required = {"year", "frequency_source_year", "imputation_method", "station_id", "station_name", "weekday_school_departures", "weekday_holiday_departures", "weekday_route_ids", "geometry"}
    missing = sorted(required - set(stops.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    stops = stops.dropna(subset=["station_id", "geometry"]).copy()
    stops["station_id"] = pd.to_numeric(stops["station_id"], errors="raise").astype("int64")
    return stops


def yearly_transport_stops(project_dir: Path, year: int) -> gpd.GeoDataFrame:
    """Read the already assigned annual stop layer produced by PT preparation."""
    stops = _read_transport(project_dir, year).copy()
    stored_years = set(pd.to_numeric(stops["year"], errors="raise").astype(int))
    if stored_years != {int(year)}:
        raise ValueError(f"Public-transport file for {year} contains years {sorted(stored_years)}")
    source_year = pd.to_numeric(stops["frequency_source_year"], errors="raise").astype(int)
    stops["pt_source_year"] = source_year
    stops["source_stop_id"] = stops["station_id"].astype(str)
    stops["source_years"] = source_year.astype(str)
    stops["source_stop_ids"] = stops["source_stop_id"]
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
    output["pt_route_ids"] = output["weekday_route_ids"].astype("string")
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
