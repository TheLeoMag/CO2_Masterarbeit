# CO2 Master Thesis

This repository contains code for my Master's thesis on regional business dynamics in Styria.

The thesis investigates how spatial location factors influence business formation and business survival. The central research question is:

> Which spatial location factors influence the settlement and survival of companies in Styria?

The analysis links company-level data from the Styrian Chamber of Commerce with open geodata, high-resolution population grids, and routing-based accessibility measures.

The empirical work is planned as a descriptive and inferential spatial analysis. 

## Repository Structure

This is a thesis analysis repository, not a standalone software package. The main folders separate source data, generated analysis data, figures, synthetic-data generation, and operational tools:

| Path | Contents |
|---|---|
| `OGD/` | Public open-government input data and processed geodata derivatives. |
| `SDG/` | Synthetic company-data generation used for the public reproducible workflow. |
| `ANAL/` | Analytical notebooks and derived 100 m panel/routing datasets. |
| `FIG/` | Figure notebooks, plotting scripts, and exported thesis figures. |
| `TOOLS/` | Local service configuration and routing support workflows. |
| `TOOLS/routing/` | Historical Valhalla routing notebooks and routing-specific documentation. |
| `TOOLS/osm-data/` | Placeholder documentation for local OpenStreetMap PBF files and derived POI catalogs. |

## Data

The following source and reference data are required for the analysis. Large geodata files are excluded from Git and must be obtained separately.

| Dataset | Purpose | Provider / source | Reference URL |
|---|---|---|---|
| Austrian OpenStreetMap extract | Building footprints, local geocoding, routing, and accessibility calculations | OpenStreetMap contributors via Geofabrik | [Geofabrik Austria](https://download.geofabrik.de/europe/austria.html) |
| Austrian 100 m population grid (POPREG) | High-resolution population distribution and population-density measures | Statistics Austria / Austrian Open Government Data | [Statistics Austria Open Data](https://data.statistik.gv.at/) |
| Styrian municipal population 2002-2025 (`Bevoelkerungsentwicklung seit 2002 in der Steiermark`) | Annual municipal population counts for backcasting the 2025 100 m population grid | Land Steiermark Open Government Data | [OGD search portal](https://app.sterz.stmk.gv.at/at.gv.stmk.aews.ext-p/p1/r/32/otogd/OGD-suche) |
| Education locations (`Bildungsstandorte`) | Higher education destinations for routing-based accessibility measures | Land Steiermark Open Government Data | [OGD search portal](https://app.sterz.stmk.gv.at/at.gv.stmk.aews.ext-p/p1/r/32/otogd/OGD-suche) |
| Public transport stops (`Haltestellen des Verkehrsverbundes Steiermark`) | Public-transport stop locations and service-frequency attributes for transit accessibility measures | Land Steiermark Open Government Data | [OGD search portal](https://app.sterz.stmk.gv.at/at.gv.stmk.aews.ext-p/p1/r/32/otogd/OGD-suche) |
| Municipality boundaries (`Gemeindegrenzen`) | Delimitation of Styria, spatial clipping, and municipality-level aggregation | Austrian Open Government Data | [OGD search portal](https://app.sterz.stmk.gv.at/at.gv.stmk.aews.ext-p/p1/r/32/otogd/OGD-suche) |
| 2025 Fachorganisation membership distribution | Reference categories and sampling weights for synthetic company data | WKO Steiermark | [Mitgliederstatistik der gewerblichen Wirtschaft Steiermark](https://www.wko.at/stmk/wirtschaft/mitgliederstatistik-der-gewerblichen-wirtschaft-steiermark) |

The GeoParquet files in `OGD/` are processed derivatives rather than separate external sources:

- `pd_popreg_100m_7767c33f-302c-11e3-beb4-0000c1ab0db6.geoparquet` is extracted from the downloaded POPREG GML using [`OGD/extract_glm.py`](OGD/extract_glm.py).
- `population_grid_styria.geoparquet` contains the population grid clipped to Styria.

## Synthetic Data

The [`SDG`](SDG/) directory contains the synthetic data generation workflow. It creates artificial company records with the same structure as the confidential source data, but does not reproduce real companies.

The generation process:

- assigns companies to economic divisions and professional groups according to the 2025 membership distribution;
- generates founding and deletion dates while keeping their chronological order valid;
- samples company locations from OpenStreetMap building footprints within Styria;
- favors commercial, office, retail, and industrial buildings, while residential and single-family homes remain possible with lower probabilities; and
- stores the resulting locations as points in EPSG:3035 and exports the dataset as GeoParquet.

Founding dates range from 1900 to 2025. Their distribution favors more recent dates while retaining a long tail of older companies. Building size also affects location selection, so larger suitable buildings can contain multiple companies.

The workflow is implemented in [`SDG/Generation.ipynb`](SDG/Generation.ipynb). It makes the public analysis reproducible without publishing sensitive company information. The final thesis analysis is performed locally using the real data.

## Notebook Order

The analysis is notebook-based. The usual public workflow is:

1. [`OGD/extract_glm.py`](OGD/extract_glm.py) converts the downloaded POPREG GML population grid to GeoParquet.
2. [`FIG/Population_Density/Population_density_Styria.ipynb`](FIG/Population_Density/Population_density_Styria.ipynb) clips the population grid to Styria and exports population-density figures.
3. [`SDG/Generation.ipynb`](SDG/Generation.ipynb) generates the synthetic company dataset.
4. [`ANAL/100m_data_foundation.ipynb`](ANAL/100m_data_foundation.ipynb) builds the 100 m raster, assigns firms to grid cells, backcasts population, and creates the quarterly panel.
5. [`TOOLS/routing/`](TOOLS/routing/) creates routing inputs, historical POI catalogs, yearly Valhalla graphs, routing features, and validation reports.
6. [`ANAL/Cox.ipynb`](ANAL/Cox.ipynb) contains the survival-analysis model work.

The routing sub-workflow has its own run order and operational notes in [`TOOLS/routing/README.md`](TOOLS/routing/README.md).

## Tools

The analysis uses:

- **Python** for data preparation, spatial analysis, and reproducible workflows.
- **Jupyter Notebook** for the main analysis steps.
- **pandas**, **GeoPandas**, **Shapely**, **pyogrio**, **pyarrow**, and **requests** for tabular data handling, geodata processing, Parquet/GeoParquet IO, OSM extraction, and local routing API calls.
- **matplotlib** for figure generation.
- **Nominatim** for local forward and reverse geocoding.
- **Valhalla** for local routing and accessibility calculations.
- **Docker Compose** to run the Nominatim and Valhalla services.
- **WSL2** with Docker available inside the configured Linux distribution for the historical Valhalla routing notebooks.

The service configuration is located in [`TOOLS/docker-compose.yml`](TOOLS/docker-compose.yml).
Before starting the Docker Compose services, download the current Austrian OpenStreetMap
extract from [Geofabrik](https://download.geofabrik.de)
and save it as `TOOLS/osm-data/austria-latest.osm.pbf`. For the historical routing
pipeline, also populate `TOOLS/osm-data/` with the yearly Austria PBF snapshots used by
the notebooks:

```text
austria-150101.osm.pbf
austria-160101.osm.pbf
austria-170101.osm.pbf
austria-180101.osm.pbf
austria-190101.osm.pbf
austria-200101.osm.pbf
austria-210101.osm.pbf
austria-220101.osm.pbf
austria-230101.osm.pbf
austria-240101.osm.pbf
austria-250101.osm.pbf
```

A placeholder file in `TOOLS/osm-data/` documents the expected filenames; `.pbf` files
are excluded from Git.

Start both services from the repository root:

```bash
docker compose -f TOOLS/docker-compose.yml up -d
```

The first startup imports the OSM data and can take a considerable amount of time.
After the import, Nominatim is available at `http://localhost:8080` and Valhalla at
`http://localhost:8002`.

## Routing

Historical routing is implemented in [`TOOLS/routing`](TOOLS/routing/). See
[`TOOLS/routing/README.md`](TOOLS/routing/README.md) for the detailed notebook order,
inputs, and output conventions. The workflow first creates active 100 m routing cells,
then builds yearly routing-destination catalogs
from historical OSM PBF snapshots plus static OGD destinations. A separate notebook builds
one Valhalla graph per year inside WSL2/Docker, and the feature-generation notebook starts
the matching yearly graph to compute nearest-destination travel times and local SLX edge
lists. Lightweight validation writes routing checks to `ANAL/data/routing/reports/`.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
