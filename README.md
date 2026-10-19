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
| Austrian 100 m population grid (POPREG)  | 2019,2025 | High-resolution population distribution | `OGD/pd_popreg_100m_*.zip` | [INSPIRE](https://geoportal.inspire.gv.at/metadatensuche/inspire/api/records/7767c33f-302c-11e3-beb4-0000c1ab0db6) ***|
| Styrian municipal population | 2002–2025 | Backcasting the 2025 population grid | `OGD/STMK_POP_2002_2025.csv` | [Land Steiermark](https://data.steiermark.at/)|
| Municipality boundaries | 2025 | Study area, clipping, and aggregation | `OGD/Gemeindegrenzen.zip` | [Land Steiermark](https://data.steiermark.at/)|
| Education locations | 2025 | Higher-education routing destinations | `OGD/Bildungsstandorte.zip` | [Land Steiermark](https://data.steiermark.at/)|
| Public-transport data | 2015–2025 | Stops, service frequencies, and rail-station destinations | `OGD/Public_Transport/` | [VERBUND Linie](https://www.verbundlinie.at/de/), [ Mobilitätsverbünde Österreich](https://data.mobilitaetsverbuende.at/de/data-sets)|
| WKO membership totals | 2025 | Sampling weights for synthetic companies | `SDG/fachorganisationen_total_members.csv` | [WKO](https://www.wko.at/stmk/wirtschaft/aktuelle-publikationen-und-statistische-daten) ****|

*GIP files for the use in this pipline need to be in the OSM PBF container format, cenverted using GIP2OSM package with the source files from GIP. 

**OpenStreetMap snapshots are not part of the routing pipeline  but are used for the building-based synthetic-data generator. In case that historic GIP files are not available teh pipline can also be run with OSM-Snapshots.

*** 2019 Dataset recovered from: [The Wayback Machine](https://web.archive.org/web/20191117185226id_/https://www.statistik.at/gs-inspire/www/inspire2/download/daten/pd_popreg_100m_7767c33f-302c-11e3-beb4-0000c1ab0db6.gml)

**** Table data extracted from PDF report.

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
8. [`ANAL/08_test_model_assumptions.ipynb`](ANAL/08_test_model_assumptions.ipynb) validates the founding and survival samples, transformations, collinearity, and event support.
9. [`ANAL/09_estimate_and_save_models.ipynb`](ANAL/09_estimate_and_save_models.ipynb) estimates the NB2 and stratified start-stop Cox main models, validates caches, and exports a portable aggregate CSV/JSON result bundle. Separate sector regressions are excluded from the standard run.
10. [`ANAL/10_visualize_model_results.ipynb`](ANAL/10_visualize_model_results.ipynb) loads a saved result bundle without raw data and creates the model figures.

Detailed routing requirements, inputs, and outputs are documented in [`ANAL/routing/README.md`](ANAL/routing/README.md).

## Resolution-specific runs

Steps 01, 03, and 06–10 have a clearly marked `CELL_SIZE` setting near the top of each notebook. Set the same integer value in every relevant notebook, restart each kernel, and run steps 01 → 03 → 06 → 07 → 08 → 09 → 10. Supported values are 100 m or larger multiples of 100 m, such as 200 m, 500 m, and 1000 m.

Each run is stored separately below `ANAL/data/<resolution>/`, for example `ANAL/data/500m/`. The shared routing destination catalogue and Valhalla road graphs remain below `ANAL/data/routing/`. Existing outputs in the former flat `ANAL/data/` layout are not reused or changed.

For analysis of already generated inputs, standalone versions of notebooks 08 and 09 are available with `100m`, `200m`, and `500m` in their filenames. Their **100 m versions explicitly use the legacy flat layout** and normalize `grid_id_100m` in memory; their 200 m and 500 m versions use the resolution folders. They do not rebuild routing. Run each notebook once and save it with its cell outputs to retain a readable record on an air-gapped computer.

Notebook 09 has one `CONFIG` cell. Model tables and compact model caches stay in the profile's internal cache directory; reuse requires matching source-content hashes, preparation code, sample, variables, fit settings, and software. Setting `run_nb2` or `run_cox` to `False` requires an existing matching model cache. The hypotheses from thesis section 3.2 are active: H1 uses the full-covariance population contrast, H2a uses the reachable-firm effect, and H2b/c use opposite expected exit-hazard signs. H3 uses the user-specified Fachorganisation classification of local demand dependence; full H4 operationalization and H5 sector heterogeneity remain open where variables or interactions are absent. H4 proxy tests and H5's direct motorway effect are labelled partial. The inference-review flags govern whether a fully operationalized hypothesis can be marked supported; partial tests do not confirm the full hypothesis. All definitions and limitations appear in `hypothesis_overview.csv` and the bottom notebook summary.

H3's main comparison is **A (high local demand dependence) versus C (low)**, with **B (mixed) kept separate**. With `run_h3=True` (default), two group count models use the same cell-period sample and allow every coefficient, period effect, and NB2 dispersion to differ. The direct difference of population effects (`mass - relative firms` in each model) uses covariance across the same grid-cell clusters, not independent-model standard errors or a comparison of two p-values. Directional influence scores are calculated in chunks and checked against each fitted marginal cluster variance; models run sequentially and reuse the main design when possible. `run_robustness=True` adds B plus separate A–B and B–C contrasts, with Holm correction for these two additional comparisons. A/C denotes the supplied demand-dependence classification, not pure service/production sector membership; the thesis interpretation must reflect that distinction.

Group counts read existing `births_by_fachgruppe_<resolution>.parquet`, or fall back to aggregating founding dates and Fachgruppen in existing firm assignments (including legacy `grid_id_100m`). The full panel supplies genuine zero observations. Unclassified births, ambiguous unsuffixed `125`/`304`, and missing source counts cause exclusion of the entire paired cell-period observation and are reported. No unknown class is silently assigned to A or C. The annual switch also controls H3 count aggregation and covariate timing. Exports include classification/coverage, group coefficients, population effects, diagnostics, and difference tests in `h3_*.csv`. H3 has its own `inference_assumptions_reviewed_h3` flag because total-count diagnostics do not establish group-model spatial or functional-form assumptions. Its internal caches include contrast influence arrays; these arrays and cell identifiers are never exported in the aggregate bundle.

Set `CONFIG["annual_counts"] = True` in any 09 version to model calendar-year founding totals instead of quarterly counts (default `False`). Only cell-years with four observed quarters are retained; births are summed before covariate filtering. Covariates use Q1 snapshots, stocks retain their existing lags, and year effects replace quarter effects. Exact calendar joins obtain previous-year population even when intermediate quarters are missing. The output reports exclusions and zero shares before/after aggregation on the same complete-year support. Annual mode does not resolve measurement-date limitations of backcast population or annual infrastructure snapshots. Cox retains quarterly start-stop intervals and event timing. Quarterly and annual caches occupy separate folders; every run has a separate bundle and records its counting interval, which notebook 10 uses for figure labels. Compare Poisson/NB2 within the same sample and counting interval; annual versus quarterly AIC is not a model ranking.

New aggregate outputs are written to `models/runs/<run_id>/`, including coefficient, hypothesis, diagnostic, comparison, sample-description, and manifest files. `latest_run.json` identifies a completed bundle; `latest_attempt.json` also records failures. Notebook 10 resolves these pointers and refuses to silently show an old run after a failed attempt. Set `LEGACY_100M=True` in notebook 10 for the legacy profile, or select `MODEL_BUNDLE_DIR` explicitly. Ordinary model errors are distinguished from grid-cluster errors; annual sector groups with very small counts are suppressed. Internal caches contain model rows and must not be included when sharing aggregate bundles.

The Cox implementation retains the existing variables and Efron partial likelihood but corrects start-stop boundary membership to `(start, stop]`. Its tie-consistent cluster-score covariance is checked against numerical case-weight derivatives; a matching library implementation is preferred. These correctness fixes can change Cox results relative to older notebooks. Spatial dependence, backcast population timing, endogeneity, and proportional-hazards screening limitations remain explicitly documented.

## Environment

Create a Python environment and install the shared dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```


## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
