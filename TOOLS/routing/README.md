# Historical Valhalla routing workflow

Run these notebooks in order:

1. `01_prepare_routing_inputs.ipynb` creates the active 100 m cell table.
2. `02_verify_yearly_routing_destinations.ipynb` checks the 2015–2025 destination files generated in `TOOLS/pois`.
3. `03_build_valhalla_graphs.ipynb` builds and smoke-tests one historical GIP graph per year.
4. `04_generate_routing_features.ipynb` generates the supported routing features.
5. `05_validate_routing_outputs.ipynb` writes `ANAL/data/routing/reports/routing_validation_summary.csv`.

All notebooks use the fixed project root `D:\CO2_Masterarbeit\CO2_Masterarbeit`. Routing destinations are `TOOLS/pois/austria-<year>-pois.geoparquet`; rail-station candidate files under `OGD/Public_Transport/Rail stations/` are diagnostic only. Destination inputs are GIP motorway exits, public-transport stops/stations, higher education, and curated centres—not OSM POIs.

## Supported outputs

- `ANAL/data/routing/inputs/active_routing_cells_100m.parquet`
- `ANAL/data/routing/features/<year>/nearest_infrastructure_100m.parquet`
- `ANAL/data/routing/features/<year>/accessibility_potentials_100m.parquet` (narrow grid-quarter population and total-firm fields)
- `ANAL/data/routing/features/<year>/fachgruppe_accessibility_quarter_100m.parquet` (long grid/year/quarter/Fachgruppe table for all 95 groups)
- `ANAL/data/routing/features/<year>/firm_accessibility_quarter_100m.parquet`

The analytical panel may retain contemporaneous `active_firms_t` for description. Model-ready routing features use only `active_firms_tminus1`; firm-level same-Fachgruppe measures apply conditional self-exclusion when the focal firm belongs to that lagged stock. Sample files, SLX edge lists, and part files are internal/legacy artifacts rather than supported outputs.
