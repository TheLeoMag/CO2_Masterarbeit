# GIP-based historical routing workflow

This directory contains analysis steps 02–07. The pipeline uses annual GIP road snapshots for both Valhalla routing and motorway-exit extraction. It does not use historical OpenStreetMap snapshots or OSM-derived POI catalogs.

## Run order

Run each notebook once, in order, from a clean kernel:

1. `02_build_yearly_destinations.ipynb`
2. `03_prepare_inputs.ipynb`
3. `04_verify_destinations.ipynb`
4. `05_build_valhalla_graphs.ipynb`
5. `06_generate_features.ipynb`
6. `07_validate_outputs.ipynb`

The notebooks locate the project root by looking for `ANAL/` and `OGD/`, so they can be launched from the repository root or a subdirectory. `destination_builders.py` contains destination-construction helpers; `routing_utils.py` contains shared output-schema helpers.

## Required inputs

- `OGD/GIP/2015.osm.pbf` through `OGD/GIP/2025.osm.pbf`
- `OGD/Public_Transport/public_transport_weekday_stop_frequency_<year>.geoparquet`
- `OGD/Gemeindegrenzen.zip`
- `OGD/Bildungsstandorte.zip`
- `SDG/companies_styria_syn.geoparquet` or the confidential local equivalent used by step 01
- outputs from `ANAL/01_build_data_foundation.ipynb`
- `ANAL/routing/static_routing_destinations.csv`

The annual destination catalogs are written to `ANAL/data/routing/destinations/`. They combine public-transport stops and rail stations, GIP-derived motorway exits, higher-education sites, and curated regional/urban centres.

## Valhalla setup

Steps 05 and 06 require WSL2 and Docker in the configured WSL distribution (`Ubuntu` by default). Import and tag the pinned Valhalla image before disconnecting an air-gapped machine:

```text
docker load -i valhalla-scripted-3.8.3.tar
docker tag <loaded-image-id> valhalla-scripted:3.8.3
```

The exact expected image name and SHA-256 ID are defined in each notebook. The notebooks create and manage year-specific containers and graphs; the removed Compose configuration is not part of this workflow.

Step 06 processes 2015–2025 with resumable checkpoints below `ANAL/data/routing/work/routing_features/<year>/`. Checkpoints are removed only after successful publication and validation. Exhausted retries leave a failure record and prevent silent publication.

## Canonical outputs

- `ANAL/data/routing/inputs/active_routing_cells_100m.parquet`
- `ANAL/data/routing/destinations/austria-<year>-pois.geoparquet`
- `ANAL/data/routing/features/<year>/nearest_infrastructure_100m.parquet`
- `ANAL/data/routing/features/<year>/accessibility_potentials_100m.parquet`
- `ANAL/data/routing/features/<year>/pedestrian_accessibility_quarter_100m.parquet`
- `ANAL/data/routing/features/<year>/fachgruppe_accessibility_quarter_100m.parquet/`
- `ANAL/data/routing/features/<year>/firm_accessibility_quarter_100m.parquet`
- `ANAL/data/routing/status/graph_build_status.csv`
- `ANAL/data/routing/status/routing_feature_status.csv`
- `ANAL/data/routing/status/accessibility_skipped_origins.csv`
- `ANAL/data/routing/reports/routing_validation_summary.csv`

Car accessibility is cumulative at 5, 10, 15, and 30 minutes. Pedestrian accessibility is cumulative at 5 and 10 minutes and includes parent-station-deduplicated stops, weekday departures, and route counts. Public-transport stops are handled by pedestrian isochrones and are not included in nearest-infrastructure routing.

Do not start model estimation until step 07 completes without a `CHECK` result.
