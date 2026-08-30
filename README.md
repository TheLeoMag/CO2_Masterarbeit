# CO2 Master Thesis

This repository contains the reproducible analysis for a Master's thesis on how spatial location factors affect business formation and survival in Styria. It combines company-level data, open geodata, a 100 m population grid, and GIP-based travel-time and accessibility measures.

## Outputs disclaimer

Generated analytical products stored below `ANAL/data/`; exported figures stored below `FIG/`. Raw `.pbf` and `.gml` files are the results of the synthetic dataset. This results are only provided for public reproducibility they do not contain or refelct real data or results.

## Repository structure

| Path | Contents |
|---|---|
| `OGD/` | External open-data inputs and source-specific preparation workflows. |
| `SDG/` | Synthetic company-data generation for the public workflow. |
| `ANAL/` | The ordered analytical workflow and generated model data. |
| `FIG/` | Descriptive and model-result figure notebooks plus exported figures. |

## Data sources

Large source files are excluded from Git and must be obtained separately.

| Dataset | Year | Purpose | Location | Source |
|---|---|---|---|---|
| GIP road-network snapshots | 2015–2025 | Valhalla graphs, travel times, road-network figures, and motorway exits | `OGD/GIP/<year>.osm.pbf` | [GIP](https://www.gip.gv.at/) *|
| Austrian OSM snapshot | 2025 | Building footprints used only to generate synthetic firm locations | `OGD/OpenStreetMap/austria-2025.osm.pbf` | [OSM](https://download.geofabrik.de/europe/austria.html#) **|
| Austrian 100 m population grid (POPREG)  | 2025 | High-resolution population distribution | `OGD/pd_popreg_100m_*.zip` | [INSPIRE](https://geoportal.inspire.gv.at/metadatensuche/inspire/api/records/7767c33f-302c-11e3-beb4-0000c1ab0db6)|
| Styrian municipal population | 2002–2025 | Backcasting the 2025 population grid | `OGD/STMK_POP_2002_2025.csv` | [Land Steiermark](https://data.steiermark.at/)|
| Municipality boundaries | 2025 | Study area, clipping, and aggregation | `OGD/Gemeindegrenzen.zip` | [Land Steiermark](https://data.steiermark.at/)|
| Education locations | 2025 | Higher-education routing destinations | `OGD/Bildungsstandorte.zip` | [Land Steiermark](https://data.steiermark.at/)|
| Public-transport data | 2015–2025 | Stops, service frequencies, and rail-station destinations | `OGD/Public_Transport/` | [VERBUND Linie](https://www.verbundlinie.at/de/)|
| WKO membership totals | 2025 | Sampling weights for synthetic companies | `SDG/fachorganisationen_total_members.csv` | [WKO](https://www.wko.at/stmk/wirtschaft/aktuelle-publikationen-und-statistische-daten) ***|

*GIP files for the use in this pipline need to be in the OSM PBF container format, cenverted using GIP2OSM package with the source files from GIP. 

**OpenStreetMap snapshots are not part of the routing pipeline  but are used for the building-based synthetic-data generator. In case that historic GIP files are not available teh pipline can also be run with OSM-Snapshots.

*** Table data extracted from PDF report.

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

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
