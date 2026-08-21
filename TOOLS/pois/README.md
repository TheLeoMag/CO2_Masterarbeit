# Yearly routing destinations (2015–2025)

This folder contains the routing-destination layers used by the historical
routing workflow.  There is one GeoParquet file per year:
`austria-<year>-pois.geoparquet`.

These are **not OSM POI extracts**.  The historical `osm-data` location was a
legacy name.  The data combines the following destination types within Styria
and a 5 km buffer:

| `poi_type` | Source |
| --- | --- |
| `pt_stop` | Public-transport stop-frequency GeoParquet |
| `rail_station` | Public-transport stops selected and clustered using the station-name rule |
| `motorway_exit` | Year-specific GIP road PBF files |
| `higher_education` | OGD education-location layer |
| `regional_centre`, `urban_centre` | Curated municipal-centre list and municipal boundaries |

`01_build_yearly_routing_destinations.ipynb` is the simple, reproducible
generator.  It uses explicit project paths and overwrites the yearly files
only when `OVERWRITE = True`.

## Required inputs

The notebook expects these existing project files:

- `TOOLS/gip-data/2015.osm.pbf` through `2025.osm.pbf` — GIP road layers used
  to derive motorway exits.
- `OGD/Public_Transport/public_transport_weekday_stop_frequency_2016.geoparquet`,
  `2017`, `2018`, `2019`, `2020`, `2022`, and `2025` — public-transport stops.
  Missing years are assigned from the nearest available year; 2021 averages
  matching 2020 and 2022 stops.
- `OGD/Gemeindegrenzen.zip` — defines the Styria study area and municipal
  representative points.
- `OGD/Bildungsstandorte.zip` — higher-education locations.
- `TOOLS/pois/static_routing_destinations.csv` — curated regional and urban
  centres.
- `TOOLS/pois/destination_builders.py` — transport and motorway-exit helper
  functions used by the notebook.

Python requirements: `geopandas`, `pandas`, `pyogrio`, `shapely`, and a
GeoParquet-capable `pyarrow` installation.

## Key fields

All files contain a point `geometry` column in EPSG:3035 plus identifiers,
name/tag fields, `year`, `poi_type`, `poi_id`, source/provenance fields,
`static_destination`, and type-specific transport/station/ramp metadata.
Public-transport stop rows also carry pipe-delimited `pt_route_ids`, allowing
distinct routes to be deduplicated across multiple reachable parent stations.
