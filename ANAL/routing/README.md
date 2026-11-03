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

Steps 03 and 06–10 have a manual `CELL_SIZE` setting. Use the same value in each of those notebooks and in step 01. Resolution-dependent routing products are written below `ANAL/data/<resolution>/routing/`; destinations and Valhalla graphs remain shared below `ANAL/data/routing/`.

## Required inputs

- `OGD/GIP/2015.osm.pbf` through `OGD/GIP/2025.osm.pbf`
- `OGD/Public_Transport/public_transport_weekday_stop_frequency_<year>.geoparquet`
- `OGD/Gemeindegrenzen.zip`
- `OGD/Bildungsstandorte.zip`
- `SDG/companies_styria_syn.geoparquet` or the confidential local equivalent used by step 01
- outputs from `ANAL/01_build_data_foundation.ipynb`
- `ANAL/routing/static_routing_destinations.csv`

The annual destination catalogs are written to `ANAL/data/routing/destinations/`. They combine all generic public-transport stops, GIP-derived motorway exits, higher-education sites, and curated regional/urban centres. Railway stations are not extracted or routed as a separate destination type; stops located at railway stations remain available as ordinary `pt_stop` records.

Step 02 defaults to `OVERWRITE=True`, so rerunning it rebuilds all requested
destination catalogs from the prepared sources. Step 06 resumes existing feature
files/checkpoints: after changing company, population or destination inputs, use
a fresh resolution-specific feature/work folder for the final run. Preserve an
earlier run in ARCHIVE if needed; resume only with the same inputs and settings.

Nearest-infrastructure travel times use a Euclidean candidate prefilter to keep
runtime feasible. They are minima among the examined destinations; the global
minimum over every destination is not guaranteed. Motorway connections represent
GIP ramp clusters with entry and/or exit connections, using representative cluster
points. The methodology draft describes this operationalisation. Disconnected
matrix batches are split so valid pairs remain available.

## Valhalla setup

Steps 05 and 06 require WSL2 and Docker in the configured WSL distribution (`Ubuntu` by default). Import and tag the pinned Valhalla image before disconnecting an air-gapped machine:

```text
docker load -i valhalla-scripted-3.8.3.tar
docker tag <loaded-image-id> valhalla-scripted:3.8.3
```

The exact expected image name and SHA-256 ID are defined in each notebook. The notebooks create and manage year-specific containers and graphs; the removed Compose configuration is not part of this workflow.

Step 06 processes 2015–2025 with resumable checkpoints below `ANAL/data/routing/work/routing_features/<year>/`. Checkpoints are removed only after successful publication and validation. Exhausted retries leave a failure record and prevent silent publication.

## Canonical outputs

- `ANAL/data/<resolution>/routing/inputs/active_routing_cells_<resolution>.parquet`
- `ANAL/data/routing/destinations/austria-<year>-pois.geoparquet`
- `ANAL/data/<resolution>/routing/features/<year>/nearest_infrastructure_<resolution>.parquet`
- `ANAL/data/<resolution>/routing/features/<year>/accessibility_potentials_<resolution>.parquet`
- `ANAL/data/<resolution>/routing/features/<year>/pedestrian_accessibility_quarter_<resolution>.parquet`
- `ANAL/data/<resolution>/routing/features/<year>/fachgruppe_accessibility_quarter_<resolution>.parquet/`
- `ANAL/data/<resolution>/routing/features/<year>/firm_accessibility_quarter_<resolution>.parquet`
- `ANAL/data/routing/status/graph_build_status.csv` (shared)
- `ANAL/data/<resolution>/routing/status/routing_feature_status.csv`
- `ANAL/data/<resolution>/routing/status/accessibility_skipped_origins.csv`
- `ANAL/data/<resolution>/routing/reports/routing_validation_summary.csv`

Car accessibility is cumulative at 5, 10, 15, and 30 minutes. Pedestrian accessibility is cumulative at 5 and 10 minutes and includes parent-station-deduplicated stops, weekday departures, and route counts. Public-transport stops are handled by pedestrian isochrones and are not included in nearest-infrastructure routing.

Do not start model estimation until step 07 completes without a `CHECK` result.

Step 07 scans Parquet with DuckDB, with a 2 GB memory limit and disk spilling if
needed. It checks exact origin/quarter and firm-event coverage, each Fachgruppe
partition, finite non-negative masses, contour nesting, route statuses and full
matching of firm and grid reference values. Documented isochrone exclusions remain
explicit. The report keeps the existing CSV format; no new dependencies are added.

Step 01 accepts complete ISO and Austrian day-first event dates. Genuine blank
dates remain missing; malformed dates, numeric serials, partial dates and reversed
founding/exit dates stop preparation. `DATE_DAYFIRST` makes the local date convention
explicit. Steps 01 and 06 use the full quarter end, including events later on its
last calendar day. Rerun 01 before generating features from changed source dates.

## Notebook progress

Notebook 06 uses a native HTML progress bar with IPython display-ID updates.
It requires no widgets, JupyterLab extensions or terminal cursor control.
Each stage retains one output showing count, percentage, elapsed time and ETA;
`PROGRESS_UPDATE_SECONDS = 2.0` limits refreshes to once every two seconds.
Resumed stages include their previous completed count and calculate their rate
from newly processed origins. Checkpoint messages remain separate.

If rich display is unavailable, the same counter falls back to a text bar logged
at most once per minute, plus start/end summaries. A display error cannot stop
routing. Copying this update requires only notebook 06; retain the real machine's
existing path, image, year and routing settings. Restart the kernel before the
next run/resume to load the new helper. Feature calculations and checkpoint
handling are unchanged.
