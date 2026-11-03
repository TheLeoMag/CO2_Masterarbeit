# CO2 Master Thesis

Analysis of spatial location factors, business formation and survival in Styria.
The public company input and its example results are synthetic, not confidential
companies or empirical thesis findings.

## Structure

| Folder | Purpose |
|---|---|
| `OGD/` | External inputs and source-specific preparation |
| `SDG/` | Public synthetic company data and generator |
| `ANAL/` | Data preparation and estimation, steps 01–09 |
| `ANAL/routing/` | Destinations, graphs and routing notebooks |
| `ANAL/data/100m/` | Canonical data for the fresh 100 m run |
| `RESULTS/100m/final/` | Seven aggregate result files |
| `FIG/` | Descriptive figures and the model-results notebook |
| `FIG/outputs/100m/final/` | Model figures and results overview |
| `ARCHIVE/` | Earlier variants and result states; ignored by Git |

Use one 08 notebook and one 09 notebook. Set CELL_SIZE consistently for another
resolution; separate notebook copies are no longer required. The old flat 100 m
layout is not selected implicitly.

## Source data

Required local sources: annual GIP PBFs for 2015–2025 under `OGD/GIP/`, observed
population grids for 2019/2025 under `OGD/Popreg_100m/`, municipal population in
`OGD/STMK_POP_2002_2025.csv`, `OGD/Gemeindegrenzen.zip`,
`OGD/Bildungsstandorte.zip`, legacy/GTFS transport sources under
`OGD/Public_Transport/`, and company records with official Sparte/Fachgruppe labels.

Sources include [GIP](https://www.gip.gv.at/),
[Statistik Austria INSPIRE](https://geoportal.inspire.gv.at/),
[Land Steiermark](https://data.steiermark.at/),
[VERBUND Linie](https://www.verbundlinie.at/de/) and
[Mobilitätsverbünde Österreich](https://data.mobilitaetsverbuende.at/de/data-sets).
Geofabrik building footprints and WKO membership totals support the public
synthetic-data generator. Large files must be obtained separately.



## Environment and verification

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests
```

Software fixtures check preparation, inference, compact exports, failure recovery
and figure generation. Confidential-data convergence and empirical assumptions
still require verification on the real-data machine.
