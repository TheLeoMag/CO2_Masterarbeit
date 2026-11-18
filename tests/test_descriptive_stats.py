"""Check original-unit recovery and exact-sample summaries on software fixtures."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "FIG/Descriptive stats")]
import descriptive_stats as ds


def make_inputs(root):
    rng = np.random.default_rng(61)
    n = 40
    people = np.tile([0., 10., 20., 30.], 10)
    firms = rng.integers(0, 20, n).astype(float)
    founding = pd.DataFrame({term: rng.normal(size=n) for term in ds.ORDER["founding"]})
    founding["log_own_pop"] = np.log1p(np.arange(n) + 10.)
    founding["population_growth_yoy"] = np.log1p(np.arange(n) + 15.) - founding.log_own_pop
    founding["log_pop_access_ring_0_15"] = np.log1p(people)
    founding["log_firms_relative_car_ring_0_15"] = np.log1p(firms) - np.log1p(people)
    founding["log_walk_pop_ring_0_10"] = np.log1p(people / 2)
    founding["log_firms_relative_walk_ring_0_10"] = np.log1p(firms / 2) - np.log1p(people / 2)
    founding["pt_ohne_haltestelle"] = np.tile([0, 0, 0, 1], 10)
    founding["walk_pt_routes_10min"] = np.arange(n) % 7
    survival = pd.DataFrame({term: rng.normal(size=n) for term in ds.ORDER["survival"]})
    survival["log_pop_ring_0_15"] = np.log1p(people)
    survival["log_same_relative_ring_0_15"] = np.log1p(firms) - np.log1p(people)
    survival["log_other_relative_ring_0_15"] = np.log1p(firms + 2) - np.log1p(people)
    survival["pt_ohne_haltestelle"] = founding.pt_ohne_haltestelle
    survival["calendar_year"] = 4
    survival["standort_id"] = [f"firm{i//2}" for i in range(n)]
    survival["start"] = np.repeat(np.arange(n//2) % 4, 2) + np.tile([0, 1], n//2)
    survival["stop"] = survival.start + 1
    survival["event"] = ((np.arange(n) % 2 == 1) & (np.arange(n) // 2 < 12)).astype(int)
    for frame in [founding, survival]:
        frame["grid_id"] = [f"grid{i//2}" for i in range(n)]
        frame["year"] = 2020
        frame["period"] = [f"2020Q{1+i%2}" for i in range(n)]
    data = root / "ANAL/data/100m"
    prepared = data / "modelldaten"
    prepared.mkdir(parents=True)
    pd.DataFrame({"grid_id": [f"grid{i}" for i in range(n//2)],
                  "municipality_id": np.where(np.arange(n//2) < 10, "60101", "60608")}).to_parquet(
                      data / "raster_100m_styria.geoparquet", index=False)
    for model, frame in [("founding", founding), ("survival", survival)]:
        frame.to_parquet(prepared / f"{model}_frame.parquet", index=False)
    meta = dict(status="completed",version="model-tables-3",resolution_m=100,
        founding_terms=ds.ORDER["founding"],survival_terms=[*ds.ORDER["survival"],"calendar_year"],
        tables={m:dict(file=f"{m}_frame.parquet",rows=n) for m in ds.MODELS})
    (prepared / "model_data_info.json").write_text(json.dumps(meta),encoding="utf-8")
    route = data / "routing/features/2020"
    route.mkdir(parents=True)
    nearest = pd.DataFrame({field:rng.uniform(1,70,n//2) for field in ds.CENTRES})
    nearest["grid_id"] = [f"grid{i}" for i in range(n//2)]
    nearest["year"] = 2020
    nearest["routing_status_urban_centre"] = "ok"
    nearest["routing_status_regional_centre"] = "ok"
    nearest.loc[0,"routing_status_urban_centre"] = "unroutable"
    nearest.to_parquet(route / "nearest_infrastructure_100m.parquet",index=False)
    bundle = root / "RESULTS/100m/final"
    bundle.mkdir(parents=True)
    info = dict(status="completed",resolution_m=100,run_name="final",created_at="2026-11-06T00:00:00+00:00",
                study_period=dict(start_year=2020,end_year=2020),
                sample_summary=dict(locations=20, exits=12, right_censored=8, left_truncated=15))
    (bundle / "laufinfo.json").write_text(json.dumps(info),encoding="utf-8")
    pd.DataFrame([dict(model=m,stage="model_sample",rows=n,units=20,events=12) for m in ds.MODELS]).to_csv(bundle / "stichprobe.csv",index=False)
    pd.DataFrame([dict(model=ds.MODEL_IDS[m],check="VIF",term=t,value=1.5+i*.1)
                  for m in ds.MODELS for i,t in enumerate(ds.ORDER[m])]).to_csv(bundle / "diagnostik.csv",index=False)
    return founding, survival, people, firms


def test_original_units_correlations_and_centres_preserve_the_full_sample(tmp_path):
    founding, survival, people, firms = make_inputs(tmp_path)
    output = tmp_path / "output"
    result = ds.analyse(tmp_path,100,"final",output)
    table = result["stats"]["founding"].set_index("Variable")
    raw = table.loc["Unternehmensbestand im Autoumfeld"]
    assert raw.Mittelwert == pytest.approx(firms.mean())
    density = table.loc["Unternehmensbestand im Autoumfeld je 1.000 Personen"]
    assert density["Ausgeschlossen (P=0)"] == 10 and density.n == 30
    assert density.Mittelwert == pytest.approx(np.mean(1000*firms[people>0]/people[people>0]))
    assert table.loc["Bevölkerungsänderung zum Vorjahr","Mittelwert"] == pytest.approx(5.)
    indicator = table.loc["Keine Haltestelle (Anteil mit Wert 1)"]
    assert indicator.Mittelwert == 25 and pd.isna(indicator.SD)
    assert table.loc["Bevölkerung im Autoumfeld (15 Minuten)","n"] == 40
    for model, frame in [("founding",founding),("survival",survival)]:
        correlation = result["correlations"][model]
        np.testing.assert_allclose(correlation.loc[ds.ORDER[model],ds.ORDER[model]],frame[ds.ORDER[model]].corr(),atol=1e-12)
    absolute = result["correlations"]["founding"]
    assert absolute.loc["log_absolute_firms_car","log_pop_access_ring_0_15"] == pytest.approx(np.corrcoef(np.log1p(firms),np.log1p(people))[0,1])
    assert "calendar_year" not in result["correlations"]["survival"]
    counts = pd.read_csv(output / "zentren_n_founding.csv",sep=";",decimal=",")
    assert counts.n.tolist() == [38,40,38,40]
    assert result["vif"].loc[result["vif"].Modell.eq("Gründung"),"VIF"].iloc[0] == 1.5
    assert result["vif"].loc[result["vif"].Modell.eq("Überleben"),"Quelle"].eq("Neu; ohne Kalendertrend").all()
    assert len(list(output.glob("*.csv"))) == 13
    km = result["kaplan_meier"]
    assert km["sample"].set_index("Gruppe").loc["Gesamt", "Linkstrunkiert"] == 15
    assert set(km["rates"].Gruppe) == set(ds.KM_GROUPS)
    assert len(km["rates"]) == 9
    for model, matrix in result["correlations"].items():
        fig = ds.heatmap(matrix,output,model,triangular=True)
        assert fig.get_size_inches()[0]*2.54 == pytest.approx(16)
        plt.close(fig)
    assert not list(output.glob("*.pdf"))
    assert "Arial" in (output / "founding.svg").read_text(encoding="utf-8")


def test_vif_matches_regression_and_exact_collinearity_is_infinite():
    rng = np.random.default_rng(13)
    values = rng.normal(size=(100,3))
    values[:,1] += values[:,0]*.8
    frame = pd.DataFrame(values,columns=list("abc"))
    result = ds.vif_from_correlation(frame.corr())
    design = np.column_stack([np.ones(len(frame)),values])
    expected = [variance_inflation_factor(design,i) for i in range(1,4)]
    np.testing.assert_allclose(result,expected)
    frame["d"] = frame.a + frame.b
    assert np.isinf(ds.vif_from_correlation(frame.corr()).loc[["a","b","d"]]).all()


def test_aggregate_only_results_and_sample_mismatch_are_explicit(tmp_path):
    make_inputs(tmp_path)
    path = tmp_path / "ANAL/data/100m/modelldaten/model_data_info.json"
    meta = path.read_text(encoding="utf-8")
    path.unlink()
    result = ds.analyse(tmp_path,100,"final",tmp_path / "output")
    assert result["stats"] == {} and result["correlations"] == {} and result["centres"] == {}
    assert not result["vif"].empty and result["missing"]
    assert result["kaplan_meier"]["sample"].empty
    path.write_text(meta,encoding="utf-8")
    frame_path = path.parent / "founding_frame.parquet"
    pd.read_parquet(frame_path).iloc[:-1].to_parquet(frame_path,index=False)
    with pytest.raises(ValueError,match="Zeilenzahl"):
        ds.analyse(tmp_path,100,"final",tmp_path / "output")


def km_locations():
    return pd.DataFrame({
        "standort_id": list("abcdef"), "entry": [0, 0, 4, 0, 0, 8],
        "exit": [4, 20, 12, 12, 24, 20], "event": [1, 0, 1, 1, 0, 1],
        "region": ["Graz"] * 3 + ["Übrige Steiermark"] * 3,
    })


def km_intervals():
    rows = []
    for location in km_locations().itertuples(index=False):
        for index, start in enumerate(range(location.entry, location.exit)):
            rows.append(dict(standort_id=location.standort_id, grid_id="grid_" + location.standort_id,
                             period=str(pd.Period("2016Q1") + index), start=start, stop=start + 1,
                             event=int(location.event and start + 1 == location.exit)))
    return pd.DataFrame(rows)


def test_km_delayed_entry_excludes_future_entrants_and_handles_tied_events_and_censoring():
    km = ds.kaplan_meier_tables(km_locations())
    rates = km["rates"].set_index(["Gruppe", "Jahre"])
    # At age 4, c enters exactly then and f enters later. The risk set is a,b,d,e.
    expected = {"Gesamt": [.75, .45, .30], "Graz": [.5, .25, .25],
                "Übrige Steiermark": [1., 2/3, 1/3]}
    for label, probabilities in expected.items():
        np.testing.assert_allclose(rates.loc[label, "Überlebensquote (%)"], 100 * np.array(probabilities))
    assert rates.loc["Gesamt", "Risikomenge"].tolist() == [4, 5, 3]
    curve = km["curves"].loc[lambda table: table.Gruppe.eq("Gesamt")].set_index("Alter (Quartale)")
    assert curve.loc[[4, 12, 20], "Risikomenge"].tolist() == [4, 5, 3]
    assert curve.loc[[4, 12, 20], "Austritte"].tolist() == [1, 2, 1]
    sample = km["sample"].set_index("Gruppe").loc["Gesamt"]
    assert (sample.Standorte, sample.Linkstrunkiert, sample.Austritte, sample.Rechtszensiert) == (6, 2, 4, 2)


def test_km_uses_complete_locations_and_the_saved_municipality_assignment(tmp_path):
    frame = km_intervals()
    grid = pd.DataFrame({"grid_id": ["grid_" + name for name in "abcdef"],
                         "municipality_id": ["60101.0"] * 3 + ["60608"] * 3})
    raster = tmp_path / "raster.parquet"
    grid.to_parquet(raster, index=False)
    with duckdb.connect() as connection:
        connection.register("model", frame)
        locations = ds.survival_locations(connection)
        locations = ds.survival_regions(connection, locations, raster)
    pd.testing.assert_frame_equal(locations.sort_values("standort_id").reset_index(drop=True),
                                  km_locations(), check_dtype=False)
    assert len(locations) == 6 < len(frame)
    km = ds.kaplan_meier_tables(locations)
    ds.kaplan_meier_plot(km["curves"], tmp_path)
    plt.close("all")
    assert (tmp_path / "figure_6_kaplan_meier.png").is_file()
    assert "Arial" in (tmp_path / "figure_6_kaplan_meier.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize("corruption", ["gap", "duplicate", "after_exit", "invalid_event"])
def test_km_rejects_invalid_histories_instead_of_changing_the_sample(corruption):
    frame = km_intervals()
    if corruption == "gap":
        frame = frame.drop(index=1)
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif corruption == "after_exit":
        frame.loc[0, "event"] = 1
    else:
        frame.loc[0, "event"] = 2
    with duckdb.connect() as connection:
        connection.register("model", frame)
        with pytest.raises(ValueError, match="Kaplan-Meier"):
            ds.survival_locations(connection)


def test_km_does_not_extrapolate_beyond_observation_or_bridge_unobserved_ages():
    locations = pd.DataFrame({"entry": [0, 8], "exit": [4, 24], "event": [0, 1]})
    km = ds.kaplan_meier_tables(locations)
    assert km["rates"]["Überlebensquote (%)"].iloc[0] == 100
    assert km["rates"]["Überlebensquote (%)"].iloc[1:].isna().all()
    assert km["curves"]["Alter (Jahre)"].max() == 1
    unsupported = ds.kaplan_meier_tables(locations.iloc[[1]])
    assert unsupported["curves"].empty
    assert unsupported["rates"]["Überlebensquote (%)"].isna().all()


def test_km_missing_entry_never_becomes_zero_and_missing_raster_retains_overall_curve(tmp_path):
    make_inputs(tmp_path)
    raster = tmp_path / "ANAL/data/100m/raster_100m_styria.geoparquet"
    raster.unlink()
    result = ds.analyse(tmp_path, 100, "final", tmp_path / "output")
    assert result["kaplan_meier"]["rates"].Gruppe.unique().tolist() == ["Gesamt"]
    assert any("raster_100m" in missing for missing in result["missing"])
    path = tmp_path / "ANAL/data/100m/modelldaten/survival_frame.parquet"
    pd.read_parquet(path).drop(columns="start").to_parquet(path, index=False)
    result = ds.analyse(tmp_path, 100, "final", tmp_path / "output")
    assert result["kaplan_meier"]["sample"].empty
    assert "fehlende Spalten: start" in (tmp_path / "output/kurzprotokoll.txt").read_text(encoding="utf-8")
    assert not list((tmp_path / "output").glob("kaplan_meier_*.csv"))


def test_descriptive_notebook_executes_the_km_section_and_exports_its_figure(tmp_path, monkeypatch):
    make_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr("IPython.display.display", lambda value: None)
    notebook = json.loads((ROOT / "FIG/Descriptive stats/Descriptive stats.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    output = tmp_path / "FIG/Descriptive stats"
    assert {"kaplan-meier-heading", "kaplan-meier-results"}.issubset({cell["id"] for cell in notebook["cells"]})
    assert (output / "figure_6_kaplan_meier.png").is_file()
    assert (output / "figure_6_kaplan_meier.svg").is_file()
    rates = pd.read_csv(output / "kaplan_meier_quoten.csv", sep=";", decimal=",")
    assert len(rates) == 9 and set(rates.Jahre) == {1, 3, 5}
