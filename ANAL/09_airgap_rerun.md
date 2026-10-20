# Notebook 09: air-gapped failure on 20 October 2026

The downloaded `09_500m_estimate_and_save_models.ipynb` has the same code as
the repository version before this fix. Its outputs establish the following:

- Input validation read and hashed 45 existing files successfully.
- Founding preparation produced 1,050,380 model rows.
- Survival joins produced 4,107,648 firm-quarter rows for 168,177 locations.
- Execution stopped in `prepare_survival()` with
  `ValueError: Missing founding dates or invalid censoring flags.`
- The subsequent `NameError: name 'survival_frame' is not defined` followed
  from continuing after that failure. Neither model had reached fitting.

The first error combines two tests, so the saved output cannot establish
whether missing founding dates, invalid exit flags, or both triggered it.
It shows a deliberate data validation failure, with no evidence of a network,
package import, optimizer, or memory failure at this point.

## Likely cause

Notebook 01 retains every firm in `firms_assigned_500m.geoparquet` and permits
missing founding dates. Routing notebook 06 explicitly excludes firms with
missing founding dates before creating firm-quarter accessibility features.
The old notebook 09 nevertheless required valid dates and flags for every
firm in the master file, including firms absent from those features. A single
undated firm outside the model sample was enough to stop the entire run.

The home file has 10,000 firms, no missing founding dates, and valid Boolean
exit flags; its routing products contain 9,751 firms. This is a different
sample from the air-gapped run, so it cannot identify the exact offending
condition in the confidential data.

## Fix prepared at home

The repository's four standalone notebook 09 variants now use pipeline
version `09-mvp-3`. Survival preparation selects the firm IDs actually present
in the study-period routing features, validates their metadata, and only then
builds the large joined interval table. Unrouted firms' missing dates do not
block the model. No routed firms are silently dropped or assigned invented
dates or censoring flags. Identifier uniqueness checks remain in place.

The printed and cached `source_date_coverage` diagnostic separates source-file
counts from routed-sample counts. Defects in the routed sample still stop
execution with specific counts, including missing metadata, missing founding
dates, invalid exit flags, missing observed exit dates, and reversed dates.

## Next air-gapped run

1. Transfer the updated `ANAL/09_500m_estimate_and_save_models.ipynb` into the
   corresponding `ANAL` directory of the air-gapped project. Use this repository
   copy, rather than the downloaded notebook containing the failed outputs.
   This notebook is self-contained; the new tests and this note are optional
   for running it. No new packages are needed for this fix.
2. Keep the existing air-gapped input files and routing products. There is no
   need to regenerate them for the validation-scope fix. Keep
   `cell_size=500`, `legacy_100m=False`, `run_nb2=True`, and `run_cox=True`.
3. Restart the kernel and run all cells from the beginning. The pipeline version
   and executed-code signatures invalidate old prepared/model caches
   automatically; deleting caches or changing `rebuild_datasets` is unnecessary.
4. Inspect `survival | source_date_coverage`. If
   `source_missing_founding_dates > 0` but all routed defect counts are zero,
   the likely original scope problem has been confirmed and preparation can
   proceed. Errors outside the sample do not alter its model rows.
5. If any routed defect count is positive, stop at the first error. Inspect and
   correct those source records inside the air gap. Do not fill founding dates
   with the analysis start date or convert unknown flags to `False`: that would
   invent firm ages or censoring. If metadata corrections change the existing
   routing histories, the affected upstream products also need to be rebuilt.
6. Save the executed notebook with its outputs. A successful end-to-end run
   must have `status: completed` in the new run's `model_run_manifest.json`.
   A new `models/latest_run.json` should point to that run. An older successful
   pointer alone does not demonstrate that this attempt succeeded.

This fix addresses the observed preparation failure. The saved notebook never
reached estimation on the confidential sample, so later convergence and model
diagnostics remain to be checked during the next run.

Validation at home: all 33 tests in `tests/test_model_mvp.py` passed, including
unrouted invalid metadata across 100/200/500 m, rejection of six kinds of routed
metadata defects, complete estimation/aggregate export, cache reuse with the
new diagnostic, and saving/reopening an executed notebook. These are synthetic
software tests, not validation of estimates on the confidential sample.
