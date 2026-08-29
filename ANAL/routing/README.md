# Historical Valhalla routing workflow (air-gapped)

Run the five notebooks once, in order, from a clean kernel:

1. `01_prepare_routing_inputs.ipynb`
2. `02_verify_yearly_routing_destinations.ipynb`
3. `03_build_valhalla_graphs.ipynb`
4. `04_generate_routing_features.ipynb`
5. `05_validate_routing_outputs.ipynb`

The notebooks discover the project root by locating `ANAL/` and `TOOLS/`. No environment variables, notebook rewriting, or orchestration scripts are required. `routing_utils.py` is the only standalone module and contains pure schema helpers.

## One-time offline setup

Transfer the source PBFs, POI files, analytical inputs, Python wheels, and the Docker image before disconnecting the PC. Import and tag the pinned image:

```text
docker load -i valhalla-scripted-3.8.3.tar
docker tag <loaded-image-id> valhalla-scripted:3.8.3
```

WSL2, Docker, and the configured WSL distribution (`Ubuntu` by default) are required. The notebooks verify image ID `sha256:1e9f511e061eefde3ebab3b860517f06e14c31a24e88403a86365e64ce6adab4`; they never pull or download anything.

Notebook 04 processes 2015–2025 sequentially with four slices and one Valhalla service. Checkpoints live only under `ANAL/data/routing/work/routing_features/<year>/`, resume after interruption, and are deleted only after successful validation. Exhausted retries leave `failed_origins.json` and prevent publication.

## Canonical outputs

- `ANAL/data/routing/inputs/active_routing_cells_100m.parquet`
- `ANAL/data/routing/features/<year>/nearest_infrastructure_100m.parquet`
- `ANAL/data/routing/features/<year>/accessibility_potentials_100m.parquet`
- `ANAL/data/routing/features/<year>/pedestrian_accessibility_quarter_100m.parquet`
- `ANAL/data/routing/features/<year>/fachgruppe_accessibility_quarter_100m.parquet/` (Hive-partitioned dataset by `Fachgruppe_ID`)
- `ANAL/data/routing/features/<year>/firm_accessibility_quarter_100m.parquet`
- `ANAL/data/routing/status/accessibility_skipped_origins.csv` (explicit exceptional origins whose routed potentials are `NaN`)

Model-ready routing features use only `active_firms_tminus1`. Car accessibility is cumulative at 5, 10, 15, and 30 minutes; pedestrian accessibility is cumulative at 5 and 10 minutes. Both retain explicit own-cell mass and reachable-cell counts. The pedestrian product stores parent-station-deduplicated PT stop counts, weekday departures, distinct normalized route IDs, and `pt_ohne_haltestelle` based on the 10-minute stop count. PT stops are no longer part of nearest-infrastructure routing. Explicitly documented car-isochrone exceptions retain their own-cell masses and pedestrian results but publish `NaN` for car-routed potentials. Notebook 05 writes `ANAL/data/routing/reports/routing_validation_summary.csv` and fails loudly if any check is not `OK`.
