# Fresh 100 m run — 3 November 2026

Use the canonical 01, routing 02–07, 08 and 09 notebooks, then FIG/10. Earlier
notebook variants and notes are preserved in ARCHIVE/2026-11-03_before_update/.

1. Set the real company input in 01, confirm membership-event field meanings
   and use CELL_SIZE=100 throughout.
2. Rebuild transport preparation and destinations with the active-reference-
   date route rule. A physical stop and measured service are distinct.
3. Reuse valid graphs with OVERWRITE=False, generate 100 m features and pass 07.
4. Run the single 08 notebook. Local model rows go to
   ANAL/data/100m/modelldaten/; no content or notebook hashes are calculated.
5. Run the single 09 notebook. Existing results with the chosen run_name are
   archived. Read RESULTS/100m/final/laufinfo.json and diagnostik.csv afterwards.
6. Run FIG/10_visualize_model_results.ipynb. Seven aggregate files supply all
   model figures, without raw-data reconstruction or model fitting.

The earlier attachments stopped in Cox fitting before export: 100 m completed
the main fit and was interrupted in its age sensitivity; 200 m did not converge
in the main fit. Safeguarded Newton steps, progress and warm starts retain the
validated likelihood/covariance. Count/main checkpoints preserve completed work.

H4 excludes station travel time. H5 includes sector effects and their joint
equality test. The main 15-minute specification stays separate from 30-minute,
centre and population-transition checks. Main and sensitivity status are
reported separately. Real-data convergence/runtime still require verification.
