# Historical Valhalla Routing Workflow

This folder contains notebook-based operational workflows for the historical Valhalla routing pipeline.

Run order:

1. `01_prepare_routing_inputs.ipynb`
2. `02_extract_yearly_osm_pois.ipynb`
3. `03_build_valhalla_graphs.ipynb`
4. `04_generate_routing_features.ipynb`
5. `05_validate_routing_outputs.ipynb`

The notebooks use the existing 100 m analytical foundation in `ANAL/data/` and the historical OSM PBF snapshots in `TOOLS/osm-data/`.

Generated yearly routing-destination files are written to `TOOLS/osm-data/austria-20xx-pois.geoparquet`. They combine historical OSM-derived POIs with static external destinations that are repeated into each year so the per-year Valhalla graphs can route them consistently. They are ignored by Git because they are derived data.

The OSM extraction follows the feature classes documented in `TOOLS/osm-data/geofabrik-osm-gis-standard-0.7.pdf`:

- motorway exits: `highway=motorway_junction`
- railway stations: `railway=station` and `railway=halt`

Static destinations appended to every yearly file:

- higher education locations from `OGD/Bildungsstandorte.zip`
- public transport stops from `OGD/Haltestellen.zip`
- regional and urban centres from `TOOLS/routing/static_routing_destinations.csv`

Notebook `04_generate_routing_features.ipynb` writes the nearest-infrastructure output and now also adds pedestrian public-transport accessibility columns to `ANAL/data/routing/features/<year>/nearest_infrastructure_100m.parquet`:

- `has_pt_stop_5min_walk`
- `pt_departures_5min_walk`

The same notebook also writes isochrone-based market-potential outputs that are independent from the SLX edge list:

- `ANAL/data/routing/features/<year>/accessibility_potentials_100m.parquet`
- `ANAL/data/routing/features/<year>/firm_accessibility_quarter_100m.parquet`

These stages use:

- `ANAL/data/raster_quarter_panel_100m.parquet` for quarter-specific population and lagged firm stocks by 100 m cell
- `ANAL/data/firms_assigned_100m.geoparquet` for the model-ready firm-quarter join
- `ANAL/data/raster_100m_styria.geoparquet` for the 100 m grid geometry and centroid-based inclusion checks

`accessibility_potentials_100m.parquet` now has one row per `grid_id`, `year`, and `quarter`, with:

- `pop_access_15min` and `pop_access_30min`
- `existing_firms_access_15min` and `existing_firms_access_30min`
- `sparte_<Sparte_ID>_access_15min` and `sparte_<Sparte_ID>_access_30min` for `Sparte_ID` 1 through 7

`firm_accessibility_quarter_100m.parquet` contains the same population and total-firm accessibility measures joined to firm-quarter rows, plus:

- `same_sector_firms_access_15min`
- `same_sector_firms_access_30min`

Self-exclusion is only applied in the firm-level file, and only when the focal firm is part of the lagged quarter stock.
