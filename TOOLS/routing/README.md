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

The same notebook also writes yearly isochrone-based market-potential outputs to `ANAL/data/routing/features/<year>/accessibility_potentials_100m.parquet`. This stage is independent from the SLX edge list and uses:

- `ANAL/data/population_backcast_100m_quarterly.parquet` for yearly backcast population values
- `ANAL/data/raster_100m_styria.geoparquet` for the 100 m grid geometry and centroid-based inclusion checks

The yearly accessibility output contains:

- `pop_access_15min`
- `pop_access_30min`
