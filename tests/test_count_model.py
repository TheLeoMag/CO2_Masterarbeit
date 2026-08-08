import numpy as np
import pandas as pd

from count_model.pipeline import ModelConfig, assert_lagged_design, estimate, lag_covariates, vif


def _panel():
    rows, rng = [], np.random.default_rng(12)
    for cell in range(12):
        for quarter in range(10):
            travel = 4 + cell / 4 + quarter / 10
            rows.append({"grid_id": f"c{cell}", "period": f"202{quarter // 4}Q{quarter % 4 + 1}", "municipality_id": f"m{cell % 3}", "births": rng.negative_binomial(2, .55), "population_backcast": 100 + cell, "active_firms_tminus1": 5, "travel_time_min": travel, "other_access": 3 + cell * cell / 40 + quarter / 7})
    return pd.DataFrame(rows)


def _config():
    return ModelConfig({"regressors": ["travel_time_min", "other_access"], "fixed_effect_level": "municipality", "offset": "population", "exposure_zero_handling": "error", "outcome_column": "births", "cluster_column": "grid_id", "panel_restriction": "test", "output_dir": "unused"})


def test_lagging_excludes_contemporaneous_values():
    frame = lag_covariates(_panel(), ["travel_time_min", "other_access"]).dropna()
    assert_lagged_design(frame, ["travel_time_min", "other_access"])
    assert "travel_time_min" not in frame
    assert frame.groupby("grid_id")["lag_travel_time_min"].count().eq(9).all()


def test_nb2_estimates_alpha_and_reports_overdispersion():
    frame = lag_covariates(_panel(), ["travel_time_min", "other_access"]).dropna()
    poisson, nb2, diagnostics = estimate(frame, _config())
    assert poisson is not None
    assert "alpha" in nb2.params.index
    assert diagnostics["alpha"] > 0
    assert diagnostics["pearson_chi2_over_df"] > 0
    assert len(vif(frame, _config())) == 2
