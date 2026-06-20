# CO2 Master Thesis

This repository contains code for my Master's thesis on regional business dynamics in Styria.

The thesis investigates how spatial location factors influence business formation and business survival. The central research question is:

> Which spatial location factors influence the settlement and survival of companies in Styria?

The analysis links company-level data from the Styrian Chamber of Commerce with open geodata, high-resolution population grids, and routing-based accessibility measures.

The empirical work is planned as a descriptive and inferential spatial analysis. 

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

## Tools

The analysis uses:

- **Python** for data preparation, spatial analysis, and reproducible workflows.
- **Nominatim** for local forward and reverse geocoding.
- **Valhalla** for local routing and accessibility calculations.
- **Docker Compose** to run the Nominatim and Valhalla services.

The service configuration is located in [`TOOLS/docker-compose.yml`](TOOLS/docker-compose.yml).
Before starting it, download the Austrian OpenStreetMap extract from
[Geofabrik](https://download.geofabrik.de/europe/austria-latest.osm.pbf) and save it as
`TOOLS/osm-data/austria-latest.osm.pbf`. A placeholder file in that directory documents
the expected filename; `.pbf` files are excluded from Git.

Start both services from the repository root:

```bash
docker compose -f TOOLS/docker-compose.yml up -d
```

The first startup imports the OSM data and can take a considerable amount of time.
After the import, Nominatim is available at `http://localhost:8080` and Valhalla at
`http://localhost:8002`.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).

