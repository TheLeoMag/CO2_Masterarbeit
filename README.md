# CO2 Master Thesis

This repository contains the reproducible analysis for a Master's thesis on how spatial location factors affect business formation and survival in Styria. It combines company-level data, open geodata, a 100 m population grid, and GIP-based travel-time and accessibility measures.

## Repository structure

| Path | Contents |
|---|---|
| `OGD/` | External open-data inputs and source-specific preparation workflows. |
| `SDG/` | Synthetic company-data generation for the public workflow. |
| `ANAL/` | The ordered analytical workflow and generated model data. |
| `ANAL/routing/` | Routing notebooks 02–07, shared helpers, and routing documentation. |
| `ANAL/data/` | Derived 100 m panels, routing products, status files, and model outputs. |
| `FIG/` | Descriptive and model-result figure notebooks plus exported figures. |
| `tests/` | Focused tests for routing feature construction. |

`TOOLS/` is no longer used. Routing, destination construction, and validation are one workflow under `ANAL/`; all external geodata now lives under `OGD/`.

## Data sources

Large source files are excluded from Git and must be obtained separately.

| Dataset | Purpose | Location |
|---|---|---|
| GIP road-network snapshots, 2015–2025 | Valhalla graphs, travel times, road-network figures, and motorway exits | `OGD/GIP/<year>.osm.pbf` |
| Austrian OSM 2025 snapshot | Building footprints used only to generate synthetic firm locations | `OGD/OpenStreetMap/austria-2025.osm.pbf` |
| Austrian 100 m population grid (POPREG) | High-resolution population distribution | `OGD/pd_popreg_100m_*.zip` |
| Styrian municipal population, 2002–2025 | Backcasting the 2025 population grid | `OGD/STMK_POP_2002_2025.csv` |
| Municipality boundaries | Study area, clipping, and aggregation | `OGD/Gemeindegrenzen.zip` |
| Education locations | Higher-education routing destinations | `OGD/Bildungsstandorte.zip` |
| Public-transport data | Stops, service frequencies, and rail-station destinations | `OGD/Public_Transport/` |
| WKO membership totals | Sampling weights for synthetic companies | `SDG/fachorganisationen_total_members.csv` |

GIP files use the OSM PBF container format, but their contents come from GIP. Historical OpenStreetMap snapshots and OSM-derived routing POIs are not part of the routing pipeline. The only remaining OSM input supports the building-based synthetic-data generator; see [`OGD/OpenStreetMap/README.md`](OGD/OpenStreetMap/README.md).

## Workflow

Prepare source-specific inputs first when they need to be regenerated:

- [`OGD/extract_glm.py`](OGD/extract_glm.py) converts the downloaded POPREG GML to GeoParquet.
- [`OGD/Public_Transport/Public_Transport_preparation.ipynb`](OGD/Public_Transport/Public_Transport_preparation.ipynb) prepares annual stop-frequency products.
- [`SDG/Generation.ipynb`](SDG/Generation.ipynb) creates the public synthetic company dataset.

Then run the numbered analysis in order:

1. [`ANAL/01_build_data_foundation.ipynb`](ANAL/01_build_data_foundation.ipynb) creates the 100 m raster, assigns firms, backcasts population, and builds the quarterly panel.
2. [`ANAL/routing/02_build_yearly_destinations.ipynb`](ANAL/routing/02_build_yearly_destinations.ipynb) combines GIP exits, public transport, higher education, and curated centres.
3. [`ANAL/routing/03_prepare_inputs.ipynb`](ANAL/routing/03_prepare_inputs.ipynb) creates active routing origins.
4. [`ANAL/routing/04_verify_destinations.ipynb`](ANAL/routing/04_verify_destinations.ipynb) validates annual destination catalogs.
5. [`ANAL/routing/05_build_valhalla_graphs.ipynb`](ANAL/routing/05_build_valhalla_graphs.ipynb) builds annual Valhalla graphs from GIP.
6. [`ANAL/routing/06_generate_features.ipynb`](ANAL/routing/06_generate_features.ipynb) generates travel-time and accessibility features.
7. [`ANAL/routing/07_validate_outputs.ipynb`](ANAL/routing/07_validate_outputs.ipynb) checks routing products before modelling.
8. [`ANAL/08_estimate_founding_model.ipynb`](ANAL/08_estimate_founding_model.ipynb) estimates the founding model.
9. [`ANAL/09_estimate_survival_model.ipynb`](ANAL/09_estimate_survival_model.ipynb) estimates the survival model.

Detailed routing requirements, inputs, and outputs are documented in [`ANAL/routing/README.md`](ANAL/routing/README.md).

## Environment

Create a Python environment and install the shared dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The routing graph and feature notebooks additionally require WSL2, Docker inside the configured WSL distribution, and the pinned Valhalla image documented in the notebooks. They start year-specific Valhalla containers themselves; no repository-level Docker Compose service is required.

Run the focused test suite from the repository root:

```bash
python -m pytest tests/test_routing_features_notebook.py
```

## Outputs and version control

Generated analytical products are stored below `ANAL/data/`; exported figures are stored below `FIG/`. Raw `.pbf` and `.gml` files, temporary routing slices, checkpoints, virtual environments, and caches are ignored. Keep confidential company data outside Git; the tracked synthetic dataset is provided for public reproducibility.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
