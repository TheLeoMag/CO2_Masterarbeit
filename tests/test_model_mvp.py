"""Behavioral checks for the standalone thesis-model notebooks (no study results)."""
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook(size=200):
    return nbformat.read(ROOT / 'ANAL' / f'09_{size}m_estimate_and_save_models.ipynb', as_version=4)


def source(nb, identifier):
    return next(c.source for c in nb.cells if c.id == identifier)


def config_source(nb, overrides):
    return source(nb, 'config').replace('PROJECT_DIR = next(', f'CONFIG.update({overrides!r})\nPROJECT_DIR = next(')


def namespace(tmp_path, monkeypatch):
    (tmp_path / 'ANAL').mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    ns = {'display': lambda value: None}
    nb = notebook()
    exec(config_source(nb, {'make_plots': False}), ns)
    for identifier in ['runtime-helpers', 'statistical-helpers', 'h3-functions', 'cox-score-helpers']:
        exec(source(nb, identifier), ns)
    return ns


def make_inputs(root, size):
    """Diverse founding cohorts and tied exits exercise the actual start-stop Cox model."""
    data = root / 'ANAL/data'
    if size != 100: data /= f'{size}m'
    data.mkdir(parents=True)
    rng = np.random.default_rng(81)
    ids = [f'software_fixture_{i:03}' for i in range(100)]
    dates = pd.period_range('2015Q1', '2025Q4', freq='Q')
    panel = pd.MultiIndex.from_product([ids, dates.astype(str)], names=['grid_id', 'period']).to_frame(index=False)
    period = pd.PeriodIndex(panel.period, freq='Q')
    panel['year'], panel['quarter'] = period.year, period.quarter
    panel['municipality_id'] = np.repeat(np.arange(100) % 5, 44).astype(str)
    panel['active_firms_tminus1'] = 1+rng.poisson(4, len(panel))
    panel['births'] = rng.negative_binomial(.8, .55, len(panel))
    panel.to_parquet(data / f'raster_quarter_panel_{size}m.parquet', index=False)
    grid = pd.DataFrame({'grid_id': ids, 'centroid_x': 4_000_000+np.arange(100)%10*size,
                         'centroid_y': 2_600_000+np.arange(100)//10*size,
                         'population_2019': rng.uniform(1, 200, 100)})
    grid['population_2025'] = grid.population_2019
    grid.to_parquet(data / f'raster_{size}m_styria.geoparquet', index=False)
    firms = pd.DataFrame({'firm_id': np.arange(100), 'grid_id': ids,
        'founding_date': pd.to_datetime([f'{2010+i%6}-01-10' for i in range(100)]),
        'exit_date': pd.to_datetime([f'{2017+i%8}-{(i%4)*3+1:02}-15' if i%3 else None for i in range(100)]),
        'exit_observed': np.arange(100)%3 != 0, 'Sparte_ID': '1', 'Sparte_Text': 'Software fixture sector'})
    main = panel[['grid_id', 'year', 'quarter', 'period']].copy()
    main['own_cell_pop'] = panel.grid_id.map(grid.set_index('grid_id').population_2019)*rng.uniform(.8, 1.2, len(panel))
    main['own_cell_firms'] = panel.active_firms_tminus1
    main['pop_access_15min'] = main.own_cell_pop+rng.gamma(4, 300, len(panel))
    main['existing_firms_access_15min'] = main.own_cell_firms+rng.poisson(25, len(panel))
    walk = main[['grid_id', 'year', 'quarter', 'period']].copy()
    walk['own_cell_walk_pop'], walk['own_cell_walk_firms'] = main.own_cell_pop, main.own_cell_firms
    walk['walk_pop_10min'] = main.own_cell_pop+rng.gamma(2, 30, len(panel))
    walk['walk_firms_10min'] = main.own_cell_firms+rng.poisson(6, len(panel))
    walk['pt_ohne_haltestelle'] = rng.integers(0, 2, len(panel))
    walk['walk_pt_routes_10min'] = np.where(walk.pt_ohne_haltestelle, 0, rng.integers(1, 10, len(panel)))
    features = main.loc[main.year.ge(2016)].copy()
    features['firm_id'] = features.grid_id.map(dict(zip(ids, range(100))))
    features['Fachgruppe_ID'] = (features.firm_id % 3).astype(str)
    features['included_in_lagged_stock'] = 1
    features['own_cell_same_fachgruppe_firms'] = 1+np.minimum(rng.poisson(2, len(features)), features.own_cell_firms.to_numpy()-1)
    features['same_fachgruppe_firms_access_15min'] = features.own_cell_same_fachgruppe_firms+rng.poisson(6, len(features))
    exits = features.firm_id.map(firms.set_index('firm_id').exit_date).dt.to_period('Q').astype('string')
    features = features.loc[exits.isna() | features.period.le(exits)]
    if size == 100:
        firms = firms.rename(columns={'grid_id': 'grid_id_100m'})
        features = features.rename(columns={'grid_id': 'grid_id_100m'})
    firms.to_parquet(data / f'firms_assigned_{size}m.geoparquet', index=False)
    for year in range(2015, 2026):
        year_dir = data / 'routing/features' / str(year)
        year_dir.mkdir(parents=True)
        main.loc[main.year.eq(year)].to_parquet(year_dir / f'accessibility_potentials_{size}m.parquet', index=False)
        walk.loc[walk.year.eq(year)].to_parquet(year_dir / f'pedestrian_accessibility_quarter_{size}m.parquet', index=False)
        nearest = grid[['grid_id']].assign(year=year, tt_motorway_exit_min=rng.uniform(1, 60, 100))
        nearest.to_parquet(year_dir / f'nearest_infrastructure_{size}m.parquet', index=False)
        if year >= 2016:
            features.loc[features.year.eq(year)].to_parquet(year_dir / f'firm_accessibility_quarter_{size}m.parquet', index=False)
    # Group births conserve each original total while containing a demand-effect difference.
    group_rng = np.random.default_rng(812)
    pop = np.log1p(main.pop_access_15min-main.own_cell_pop)
    probability_a = .8/(1+np.exp(-.7*(pop-pop.mean())/pop.std()))
    birth_rows = []
    for i, row in panel.iterrows():
        counts = group_rng.multinomial(int(row.births), [probability_a.iloc[i], .2, .8-probability_a.iloc[i]])
        for code, count in zip(['118', '101', '201'], counts):
            if count:
                birth_rows.append({'grid_id': row.grid_id, 'period': row.period, 'Fachgruppe_ID': code, 'births': count})
    pd.DataFrame(birth_rows).to_parquet(data / f'births_by_fachgruppe_{size}m.parquet', index=False)


def execute(root, size, monkeypatch, overrides=None, edit_preparation=False):
    monkeypatch.chdir(root)
    nb = notebook(size)
    ns = {'display': lambda value: None}
    settings = {'make_plots': False, 'moran_permutations': 19}
    settings.update(overrides or {})
    for c in nb.cells:
        if c.cell_type == 'code':
            code = config_source(nb, settings) if c.id == 'config' else c.source
            if edit_preparation and c.id=='runtime-helpers':
                code = code.replace('return difference.clip(lower=0)', 'return difference.clip(lower=0) + .1')
            exec(compile(code, c.id, 'exec'), ns)
    return ns


def test_all_notebooks_compile_and_are_self_contained():
    for path in [*ROOT.joinpath('ANAL').glob('09*.ipynb'), ROOT/'ANAL/10_visualize_model_results.ipynb']:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type=='code':
                compile(c.source, str(path)+'/'+c.id, 'exec')
                assert not c.outputs


def survival_preparation_namespace(tmp_path, size, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nb = notebook(size)
    ns = {'display': lambda value: None}
    exec(config_source(nb, {'make_plots': False}), ns)
    for identifier in ['runtime-helpers', 'prepare-founding-function', 'prepare-survival-function']:
        exec(source(nb, identifier), ns)
    return ns


@pytest.mark.parametrize('size', [100, 200, 500])
def test_unrouted_invalid_metadata_does_not_block_survival(size, tmp_path, monkeypatch):
    make_inputs(tmp_path, size)
    ns = survival_preparation_namespace(tmp_path, size, monkeypatch)
    expected = ns['prepare_survival']()
    firms = pd.read_parquet(ns['FIRMS_PATH'])
    extra = firms.iloc[[0]].copy()
    extra['firm_id'] = 999
    extra['founding_date'] = pd.NaT
    # Invalid metadata outside routing must not change the model sample.
    extra['exit_observed'] = 2
    firms['exit_observed'] = firms.exit_observed.astype(int)
    pd.concat([firms, extra], ignore_index=True).to_parquet(ns['FIRMS_PATH'], index=False)
    actual = ns['prepare_survival']()
    pd.testing.assert_frame_equal(actual, expected)
    audit = json.loads(next(row['detail'] for row in reversed(ns['diagnostic_rows'])
                            if row['check'] == 'source_date_coverage'))
    assert audit['source_firms'] == 101
    assert audit['unrouted_source_firms'] == 1
    assert audit['source_missing_founding_dates'] == 1
    assert audit['source_invalid_exit_flags'] == 1
    assert audit['routed_missing_founding_dates'] == 0
    assert audit['routed_invalid_exit_flags'] == 0
    ns['con'].close()


@pytest.mark.parametrize(('defect', 'message'), [
    ('missing_founding', 'missing_founding_dates=1; invalid_exit_flags=0'),
    ('invalid_flag', 'missing_founding_dates=0; invalid_exit_flags=1'),
    ('missing_flag', 'missing_founding_dates=0; invalid_exit_flags=1'),
    ('missing_exit', 'observed_exits_without_dates=1; exits_before_founding=0'),
    ('reversed_dates', 'observed_exits_without_dates=0; exits_before_founding=1'),
    ('missing_metadata', 'Routed firms missing source metadata: firms=1'),
])
def test_invalid_routed_metadata_still_fails_before_interval_join(defect, message, tmp_path, monkeypatch, capsys):
    make_inputs(tmp_path, 500)
    ns = survival_preparation_namespace(tmp_path, 500, monkeypatch)
    firms = pd.read_parquet(ns['FIRMS_PATH'])
    if defect == 'missing_founding':
        firms.loc[1, 'founding_date'] = pd.NaT
    elif defect in ['invalid_flag', 'missing_flag']:
        firms['exit_observed'] = firms.exit_observed.astype(float)
        firms.loc[1, 'exit_observed'] = 2. if defect == 'invalid_flag' else np.nan
    elif defect == 'missing_exit':
        firms.loc[1, 'exit_date'] = pd.NaT
    elif defect == 'reversed_dates':
        firms.loc[1, 'exit_date'] = pd.Timestamp('2000-01-01')
    else:
        firms = firms.drop(index=1)
    firms.to_parquet(ns['FIRMS_PATH'], index=False)
    with pytest.raises(ValueError, match=message):
        ns['prepare_survival']()
    assert ns['diagnostic_rows'][-1]['status'] == 'FAIL'
    assert 'Joined survival data:' not in capsys.readouterr().out
    ns['con'].close()


def test_unrouted_missing_dates_pipeline_and_cached_audit(tmp_path, monkeypatch):
    make_inputs(tmp_path, 500)
    path = tmp_path/'ANAL/data/500m/firms_assigned_500m.geoparquet'
    firms = pd.read_parquet(path)
    extra = firms.iloc[[0]].copy()
    extra['firm_id'], extra['founding_date'] = 999, pd.NaT
    pd.concat([firms, extra], ignore_index=True).to_parquet(path, index=False)
    first = execute(tmp_path, 500, monkeypatch)
    assert first['manifest']['status'] == 'completed'
    second = execute(tmp_path, 500, monkeypatch, {'run_nb2': False, 'run_cox': False})
    assert second['cox_fit']['loaded_from_cache']
    manifest = json.loads((second['BUNDLE_DIR']/'model_run_manifest.json').read_text(encoding='utf-8'))
    audit = next(row for row in manifest['diagnostics'] if row['check'] == 'source_date_coverage')
    assert json.loads(audit['detail'])['source_missing_founding_dates'] == 1
    assert json.loads(audit['detail'])['routed_missing_founding_dates'] == 0


def test_contrasts_use_full_covariance_and_joint_tests_have_no_fake_effect(tmp_path, monkeypatch):
    ns = namespace(tmp_path, monkeypatch)
    fit = {'names': ['a', 'b'], 'params': [1., .4], 'covariance': [[.09, .03], [.03, .04]]}
    value, se = ns['linear_effect'](fit, {'a': 1, 'b': -1})
    assert value == pytest.approx(.6)
    assert se**2 == pytest.approx(.09+.04-2*.03)
    defs = [{'enabled': True, 'hypothesis_id': 'TEST', 'description': 'Software test', 'test_type': 'joint',
             'terms': ['a', 'b'], 'expected_direction': 'either'}]
    table = ns['hypothesis_table'](fit, defs, False)
    assert pd.isna(table.loc[0, 'estimate'])
    assert table.loc[0, 'supported'] is None
    assert 'irr' not in table


def test_efron_cluster_scores_and_split_intervals_are_consistent(tmp_path, monkeypatch):
    ns = namespace(tmp_path, monkeypatch)
    assert isinstance(ns['verify_cox_cluster_implementation'](), bool)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 2))
    stop = np.repeat(np.arange(2, 12), 2).astype(float)
    status = np.arange(20)%3 != 0
    beta = np.array([.1, -.2])
    PHReg = ns['PHReg']
    first = PHReg(stop, x, status=status, entry=np.nextafter(np.zeros(20), np.inf), ties='efron')
    # Split at an actual tied event time; only one interval per firm may be at risk there.
    split_x, starts, stops, events, groups = [], [], [], [], []
    for i in range(20):
        if stop[i] > 2:
            split_x.extend([x[i], x[i]]); starts.extend([0, 2]); stops.extend([2, stop[i]])
            events.extend([False, status[i]]); groups.extend([i, i])
        else:
            split_x.append(x[i]); starts.append(0); stops.append(stop[i]); events.append(status[i]); groups.append(i)
    second = PHReg(np.asarray(stops), np.asarray(split_x), status=np.asarray(events), entry=np.nextafter(np.asarray(starts, dtype=float), np.inf), ties='efron')
    assert first.loglike(beta) == pytest.approx(second.loglike(beta), abs=1e-10)
    np.testing.assert_allclose(first.score(beta), second.score(beta), atol=1e-10)
    one = ns['cox_score_blocks'](first, beta, np.arange(20))[0]
    two = ns['cox_score_blocks'](second, beta, groups)[0]
    np.testing.assert_allclose(one, two, atol=1e-10)


@pytest.mark.parametrize('size', [100, 200, 500])
def test_full_pipeline_exports_only_aggregates(size, tmp_path, monkeypatch):
    make_inputs(tmp_path, size)
    ns = execute(tmp_path, size, monkeypatch)
    assert ns['manifest']['status']=='completed'
    assert ns['founding_fit']['converged'] and ns['cox_fit']['converged']
    assert ns['founding_fit']['family']=='nb2'
    assert bool(ns['compatibility_mappings']) == (size==100)
    assert not ns['CONFIG']['run_sector_models']
    assert set(ns['founding_hypotheses'].hypothesis_id) == {
        'H1', 'H2a', 'H3', 'H4', 'H4_routes', 'H4_no_stop', 'H4_joint', 'H5', 'H5_heterogeneity'}
    assert set(ns['cox_hypotheses'].hypothesis_id) == {'H2b', 'H2c'}
    assert ns['founding_hypotheses'].set_index('hypothesis_id').loc['H3', 'coverage'] == 'tested'
    assert ns['manifest']['h3']['status'] == 'estimated'
    assert set(ns['h3_fits']) == {'A', 'C'}
    assert ns['h3_tests'].loc[0, 'estimate'] == pytest.approx(
        ns['h3_fits']['A']['population_effect']['estimate']-ns['h3_fits']['C']['population_effect']['estimate'])
    assert ns['cox_fit']['covariance_method'] != 'nonrobust'
    bundle = ns['BUNDLE_DIR']
    required = ['founding_nb2_coefficients.csv', 'founding_nb2_hypotheses.csv', 'founding_nb2_diagnostics.csv',
                'survival_cox_coefficients.csv', 'survival_cox_hypotheses.csv', 'survival_cox_diagnostics.csv',
                'sample_description.csv', 'model_comparison.csv', 'model_run_manifest.json']
    assert all((bundle/name).is_file() for name in required)
    for path in bundle.glob('*.csv'):
        frame = pd.read_csv(path)
        assert not {'grid_id', 'firm_id', 'standort_id'}.intersection(frame.columns), path.name
    original_predictions = np.exp(ns['X_founding'].to_numpy() @ ns['beta_count'].to_numpy())
    np.testing.assert_allclose(original_predictions, ns['mu_founding'])
    manifest = json.loads((bundle/'model_run_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['datasets'])==45
    assert manifest['survival']['entry_convention'].startswith('(start, stop]')
    overview = pd.read_csv(bundle/'hypothesis_overview.csv')
    assert len(overview) == 11
    assert overview.loc[overview.coverage.eq('unavailable'), 'p_value'].isna().all()


def test_cache_reuse_invalidation_and_failed_latest_attempt(tmp_path, monkeypatch):
    make_inputs(tmp_path, 200)
    first = execute(tmp_path, 200, monkeypatch)
    second = execute(tmp_path, 200, monkeypatch, {'run_nb2': False, 'run_cox': False})
    assert second['founding_fit']['loaded_from_cache'] and second['cox_fit']['loaded_from_cache']
    assert first['founding_fit']['params'] == second['founding_fit']['params']
    panel = second['PANEL_PATH']
    frame = pd.read_parquet(panel)
    frame.loc[frame.year.eq(2018).idxmax(), 'births'] += 1
    frame.to_parquet(panel, index=False)
    with pytest.raises(RuntimeError, match='no exactly matching model cache'):
        execute(tmp_path, 200, monkeypatch, {'run_nb2': False, 'run_cox': False})
    attempt = json.loads((second['RESULT_DIR']/'latest_attempt.json').read_text(encoding='utf-8'))
    assert attempt['status']=='failed'
    reader = nbformat.read(ROOT/'ANAL/10_visualize_model_results.ipynb', as_version=4)
    ns = {'display': lambda value: None}
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    exec(reader.cells[2].source.replace('CELL_SIZE=100', 'CELL_SIZE=200'), ns)
    with pytest.raises(RuntimeError, match='latest model attempt'):
        exec(reader.cells[4].source, ns)


def test_nonconverged_nb2_is_not_published_as_success(tmp_path, monkeypatch):
    make_inputs(tmp_path, 500)
    with pytest.raises(RuntimeError, match='did not converge'):
        execute(tmp_path, 500, monkeypatch, {'maxiter_nb2': 1})
    result_dir = tmp_path/'ANAL/data/500m/models'
    attempt = json.loads((result_dir/'latest_attempt.json').read_text(encoding='utf-8'))
    assert attempt['status']=='failed'
    bundle = result_dir/attempt['bundle']
    assert (bundle/'founding_poisson_robust_alternative.csv').exists()
    assert not (bundle/'founding_nb2_coefficients.csv').exists()
    assert not (result_dir/'latest_run.json').exists()


def test_optional_bounded_fits_and_successful_plot_bundle_selection(tmp_path, monkeypatch):
    make_inputs(tmp_path, 200)
    ns = execute(tmp_path, 200, monkeypatch, {'run_robustness': True, 'run_offset': True})
    assert len(ns['robustness_table'])==3
    assert set(ns['h3_fits']) == {'A', 'B', 'C'}
    assert set(ns['h3_tests'].hypothesis_id) == {'H3', 'H3_B_vs_C', 'H3_A_vs_B'}
    assert ns['h3_tests'].loc[ns['h3_tests'].hypothesis_id.ne('H3'), 'p_holm_robustness'].notna().all()
    assert ns['h3_tests'].loc[ns['h3_tests'].hypothesis_id.eq('H3'), 'p_holm_robustness'].isna().all()
    assert ns['cox_fit']['iterations'] <= ns['CONFIG']['maxiter_cox']
    reader = nbformat.read(ROOT/'ANAL/10_visualize_model_results.ipynb', as_version=4)
    plotting = {'display': lambda value: None}
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    exec(reader.cells[2].source.replace('CELL_SIZE=100', 'CELL_SIZE=200'), plotting)
    exec(reader.cells[4].source, plotting)
    assert plotting['RESULT_DIR']==ns['BUNDLE_DIR']
    assert plotting['FOUNDING_LABEL']=='NB2'


def test_explicit_poisson_alternative_and_disabled_diagnostics(tmp_path, monkeypatch):
    make_inputs(tmp_path, 500)
    ns = execute(tmp_path, 500, monkeypatch, {'founding_model': 'poisson_robust', 'run_diagnostics': False})
    assert ns['founding_fit']['family']=='poisson'
    assert not (ns['BUNDLE_DIR']/'founding_nb2_coefficients.csv').exists()
    assert (ns['BUNDLE_DIR']/'founding_poisson_coefficients.csv').exists()
    assert any(row['status']=='OFFEN' for row in ns['diagnostic_rows'])
    reader = nbformat.read(ROOT/'ANAL/10_visualize_model_results.ipynb', as_version=4)
    plotting = {'display': lambda value: None}
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    exec(reader.cells[2].source.replace('CELL_SIZE=100', 'CELL_SIZE=500'), plotting)
    exec(reader.cells[4].source, plotting)
    assert plotting['FOUNDING_PREFIX']=='founding_poisson'


def test_saved_executed_notebook_retains_results(tmp_path):
    from nbclient import NotebookClient
    make_inputs(tmp_path, 100)
    nb = notebook(100)
    next(c for c in nb.cells if c.id=='config').source = config_source(nb, {'make_plots': False, 'moran_permutations': 19})
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(tmp_path)}}).execute()
    saved = tmp_path/'executed_legacy_100m.ipynb'
    nbformat.write(nb, saved)
    reopened = nbformat.read(saved, as_version=4)
    assert reopened.cells[-1].id=='summary'
    for identifier in ['fit-count', 'fit-cox', 'export-results', 'summary']:
        cell = next(c for c in reopened.cells if c.id==identifier)
        assert cell.outputs
        assert not any(out.output_type=='error' for out in cell.outputs)


def test_editing_preparation_code_invalidates_cache_without_changing_column_names(tmp_path, monkeypatch):
    make_inputs(tmp_path, 200)
    execute(tmp_path, 200, monkeypatch)
    with pytest.raises(RuntimeError, match='no exactly matching model cache'):
        execute(tmp_path, 200, monkeypatch, {'run_nb2': False, 'run_cox': False}, edit_preparation=True)


def test_user_hypotheses_use_absolute_stock_contrasts_and_exit_hazard_signs(tmp_path, monkeypatch):
    ns = namespace(tmp_path, monkeypatch)
    names = ['log_pop_access_ring_0_15', 'log_firms_relative_car_ring_0_15',
             'walk_pt_routes_10min', 'pt_ohne_haltestelle', 'log_tt_motorway_exit']
    cov = np.eye(5)*.01
    cov[0, 1] = cov[1, 0] = .004
    fit = {'names': names, 'params': [1., .2, .5, -.4, -.3], 'covariance': cov.tolist()}
    table = ns['hypothesis_table'](fit, ns['HYPOTHESES_NB2'], True).set_index('hypothesis_id')
    assert table.loc['H1', 'estimate'] == pytest.approx(.8)
    assert table.loc['H1', 'std_error']**2 == pytest.approx(.01+.01-2*.004)
    assert table.loc['H2a', 'estimate'] == pytest.approx(.2)
    assert table.loc['H1', 'supported']
    for identifier in ['H3', 'H4', 'H5_heterogeneity']:
        assert table.loc[identifier, 'coverage'] == 'unavailable'
        assert pd.isna(table.loc[identifier, 'p_value'])
        assert table.loc[identifier, 'supported'] is None
    for identifier in ['H4_routes', 'H4_no_stop', 'H4_joint', 'H5']:
        assert table.loc[identifier, 'coverage'] == 'partial'
        assert table.loc[identifier, 'supported'] is None
    cox_fit = {'names': ['log_same_relative_ring_0_15', 'log_other_relative_ring_0_15'],
               'params': [.4, -.5], 'covariance': [[.01, .002], [.002, .01]]}
    cox = ns['hypothesis_table'](cox_fit, ns['HYPOTHESES_COX'], True).set_index('hypothesis_id')
    assert cox.loc['H2b', 'expected_direction'] == 'positive'
    assert cox.loc['H2c', 'expected_direction'] == 'negative'
    assert cox.supported.all()
    missing = ns['hypothesis_table'](fit, ns['HYPOTHESES_COX'], True)
    assert missing.coverage.eq('unavailable').all()
    assert missing.p_value.isna().all()


@pytest.mark.parametrize('size', [100, 200, 500])
def test_annual_counts_sum_original_quarters_and_do_not_average_covariates(size, tmp_path, monkeypatch):
    make_inputs(tmp_path, size)
    monkeypatch.chdir(tmp_path)
    nb = notebook(size)
    ns = {'display': lambda value: None}
    exec(config_source(nb, {'annual_counts': True, 'make_plots': False}), ns)
    exec(source(nb, 'runtime-helpers'), ns)
    exec(source(nb, 'prepare-founding-function'), ns)
    panel = pd.read_parquet(ns['PANEL_PATH'])
    # One missing observed quarter must exclude a whole cell-year, not create a partial-year sum.
    panel = panel.loc[~(panel.grid_id.eq('software_fixture_000') & panel.year.eq(2018) & panel.quarter.eq(2))]
    # A wholly absent cell-year also belongs in exclusions, with unknown rather than invented births.
    panel = panel.loc[~(panel.grid_id.eq('software_fixture_002') & panel.year.eq(2020))]
    panel.to_parquet(ns['PANEL_PATH'], index=False)
    # Q2 covariates are deliberately irrelevant to the Q1-snapshot annual specification.
    walk_path = ns['FEATURE_DIR']/'2019'/f'pedestrian_accessibility_quarter_{size}m.parquet'
    walk = pd.read_parquet(walk_path)
    walk.loc[walk.quarter.ne(1), 'walk_pop_10min'] = np.nan
    walk.to_parquet(walk_path, index=False)
    frame = ns['prepare_founding']()
    expected = panel.loc[panel.year.ge(2016)].groupby(['grid_id', 'year']).births.agg(['sum', 'size'])
    expected = expected.loc[expected['size'].eq(4), 'sum'].sort_index()
    # 2021 additionally lacks the previous-year Q1 population because all of 2020 is absent.
    expected = expected.drop(('software_fixture_002', 2021))
    pd.testing.assert_series_equal(frame.set_index(['grid_id', 'year']).births.sort_index(), expected, check_names=False)
    assert frame.period.eq(frame.year.astype(str)).all()
    assert frame.quarter.eq(1).all()
    assert len(frame) == 997
    assert not ((frame.grid_id == 'software_fixture_000') & (frame.year == 2018)).any()
    row = frame.loc[frame.grid_id.eq('software_fixture_001') & frame.year.eq(2019)].iloc[0]
    quarter_start = panel.loc[panel.grid_id.eq(row.grid_id) & panel.year.eq(2019) & panel.quarter.eq(1)].iloc[0]
    assert row.log_own_firms == pytest.approx(np.log1p(quarter_start.active_firms_tminus1))
    car_previous = pd.read_parquet(ns['FEATURE_DIR']/'2018'/f'accessibility_potentials_{size}m.parquet')
    previous_pop = car_previous.loc[car_previous.grid_id.eq(row.grid_id) & car_previous.quarter.eq(1), 'own_cell_pop'].iloc[0]
    assert row.log_own_pop == pytest.approx(np.log1p(previous_pop))
    stages = pd.DataFrame(ns['sample_rows']).set_index('stage')
    assert stages.loc['annual_complete_years', 'excluded'] == 2
    assert 'entirely missing cell-years=1' in stages.loc['annual_complete_years', 'detail']
    assert stages.loc['model_sample', 'excluded'] == 1  # Missing lag, after two coverage exclusions.
    assert stages.loc['annual_complete_years', 'events'] == stages.loc['annual_complete_quarters', 'events']
    assert stages.loc['annual_complete_years', 'zero_share'] <= stages.loc['annual_complete_quarters', 'zero_share']


def test_annual_pipeline_cache_isolation_year_effects_and_reader_labels(tmp_path, monkeypatch):
    make_inputs(tmp_path, 200)
    quarterly = execute(tmp_path, 200, monkeypatch)
    with pytest.raises(RuntimeError, match='no exactly matching model cache'):
        execute(tmp_path, 200, monkeypatch, {'annual_counts': True, 'run_nb2': False, 'run_cox': False})
    annual = execute(tmp_path, 200, monkeypatch, {'annual_counts': True})
    assert annual['CACHE_DIR'] != quarterly['CACHE_DIR']
    assert len(annual['founding']) == len(quarterly['founding'])//4
    assert annual['y_founding'].sum() == quarterly['y_founding'].sum()
    assert all('Q' not in t for t in annual['X_founding'].columns if t.startswith('period_'))
    assert len([t for t in annual['X_founding'] if t.startswith('period_')]) == 9
    pd.testing.assert_frame_equal(annual['survival_frame'], quarterly['survival_frame'])
    q1 = quarterly['founding'].loc[quarterly['founding'].quarter.eq(1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(annual['founding'][annual['FOUNDING_TERMS']], q1[annual['FOUNDING_TERMS']])
    assert set(annual['moran_table'].period) == {'2016', '2025'}
    assert annual['manifest']['time_aggregation']['founding'] == 'year'
    assert annual['manifest']['time_aggregation']['survival'] == 'quarter'
    reader = nbformat.read(ROOT/'ANAL/10_visualize_model_results.ipynb', as_version=4)
    plotting = {'display': lambda value: None}
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    exec(reader.cells[2].source.replace('CELL_SIZE=100', 'CELL_SIZE=200'), plotting)
    exec(reader.cells[4].source, plotting)
    assert plotting['FOUNDING_FREQUENCY'] == 'year'
    assert plotting['RESULT_DIR'] == annual['BUNDLE_DIR']
    plotting['plt'].switch_backend('Agg')
    for cell_index in [10, 12]:
        exec(reader.cells[cell_index].source, plotting)
    plotting['plt'].close('all')
    reused = execute(tmp_path, 200, monkeypatch, {'annual_counts': True, 'run_nb2': False, 'run_cox': False})
    assert reused['founding_fit']['loaded_from_cache']
    assert reused['cox_fit']['loaded_from_cache']
    assert all(fit['loaded_from_cache'] for fit in reused['h3_fits'].values())
    assert any(row['stage'] == 'annual_complete_years' for row in reused['sample_rows'])
    quarterly_reused = execute(tmp_path, 200, monkeypatch, {'run_nb2': False, 'run_cox': False})
    assert quarterly_reused['founding_fit']['params'] == quarterly['founding_fit']['params']


def test_h3_classification_preserves_suffixes_and_does_not_guess_unknown_codes(tmp_path, monkeypatch):
    ns = namespace(tmp_path, monkeypatch)
    codes = ns['normalize_fachgruppe'](pd.Series(['118.0', ' 125a ', '125B', '304 A', '304b', '125', '304', '299', None]))
    groups = codes.map(ns['H3_GROUP_MAP'])
    assert groups.iloc[:5].tolist() == ['A', 'A', 'A', 'B', 'C']
    assert groups.iloc[5:].isna().all()
    assert len(ns['H3_CLASSIFICATION_ROWS']) == len(ns['H3_GROUP_MAP'])
    assert all(ns['H3_GROUP_MAP'][str(i)] == 'C' for i in range(201, 208))
    assert ns['H3_GROUP_MAP']['209'] == 'B'
    assert all(ns['H3_GROUP_MAP'][str(i)] == 'B' for i in range(401, 407))


def test_h3_influence_matches_library_scores_and_uses_cross_model_covariance(tmp_path, monkeypatch):
    ns = namespace(tmp_path, monkeypatch)
    rng = np.random.default_rng(19)
    x = pd.DataFrame(rng.normal(size=(50003, 2)), columns=['mass', 'relative'])
    y = rng.negative_binomial(2, .6, len(x)).astype(float)
    scale = pd.Series([1.3, 2.5], index=x.columns)
    fit = {'family': 'nb2', 'names': ['mass', 'relative', 'alpha'], 'params': [.2, -.1, .6],
           'naive_covariance': [[.2, .01, .02], [.01, .3, .05], [.02, .05, .1]]}
    weights = {'mass': 1, 'relative': -1}
    calculated = ns['h3_directional_scores'](x, y, fit, scale, weights)
    model = ns['NegativeBinomial'](y, x, loglike_method='nb2', check_rank=False)
    model._transparams = False
    factor = np.r_[scale.to_numpy(), 1.]
    direction = factor*(np.asarray(fit['naive_covariance']) @ np.array([1., -1., 0.]))
    expected = model.score_obs(np.asarray(fit['params'])*factor) @ direction
    np.testing.assert_allclose(calculated, expected, rtol=1e-8, atol=1e-9)
    first = {'population_effect': {'estimate': .4, 'std_error': .1}, 'population_cluster_influence': [.1, -.2, .15]}
    second = {'population_effect': {'estimate': .1, 'std_error': .1}, 'population_cluster_influence': [.05, -.1, -.2]}
    result = ns['h3_difference'](first, second, 'H3', 'positive')
    expected_variance = np.sum((np.array(first['population_cluster_influence'])-second['population_cluster_influence'])**2)
    assert result['estimate'] == pytest.approx(.3)
    assert result['std_error']**2 == pytest.approx(expected_variance)
    assert result['cross_group_covariance'] == pytest.approx(np.dot(first['population_cluster_influence'], second['population_cluster_influence']))
    assert result['supported'] is None


@pytest.mark.parametrize('annual', [False, True])
def test_h3_counts_include_zeros_conserve_births_and_exclude_unknown_paired_observations(annual, tmp_path, monkeypatch):
    make_inputs(tmp_path, 100)
    monkeypatch.chdir(tmp_path)
    nb = notebook(100)
    ns = {'display': lambda value: None}
    exec(config_source(nb, {'annual_counts': annual, 'make_plots': False}), ns)
    for identifier in ['runtime-helpers', 'prepare-founding-function', 'h3-functions']:
        exec(source(nb, identifier), ns)
    ns['founding'] = ns['prepare_founding']()
    sparse = pd.read_parquet(ns['H3_BIRTHS_PATH'])
    target = sparse.loc[sparse.period.eq('2018Q2')].index[0]
    target_grid, unknown_births = sparse.loc[target, ['grid_id', 'births']]
    sparse.loc[target, 'Fachgruppe_ID'] = '304'  # Ambiguous split: never silently classified.
    sparse = sparse.rename(columns={'grid_id': 'grid_id_100m'})
    sparse.to_parquet(ns['H3_BIRTHS_PATH'], index=False)
    counts, keep, coverage, report = ns['prepare_h3_counts']()
    np.testing.assert_array_equal(counts.sum(axis=1).to_numpy(), ns['founding'].births.to_numpy())
    target_period = '2018' if annual else '2018Q2'
    assert counts.loc[(target_grid, target_period), 'UNCLASSIFIED'] == unknown_births
    assert report['rows_excluded_unknown'] == 1
    assert not keep[counts.index.get_loc((target_grid, target_period))]
    assert report['unclassified_births'] == unknown_births
    assert coverage.loc[coverage.Fachgruppe_ID.eq('304'), 'demand_group'].eq('UNCLASSIFIED').all()
    assert (counts.loc[keep, ['A', 'B', 'C']] == 0).any().any()  # Positive-only source yields real zero panel counts.
    # A missing classified source row is also unknown rather than a fabricated zero group count.
    removed = sparse.loc[sparse.period.eq('2019Q3')].index[0]
    sparse = sparse.drop(index=removed)
    sparse.to_parquet(ns['H3_BIRTHS_PATH'], index=False)
    _, _, coverage, report = ns['prepare_h3_counts']()
    assert report['rows_excluded_unknown'] == 2
    assert coverage.Fachgruppe_ID.eq('<missing from source>').any()


def test_h3_legacy_fallback_reuses_existing_firm_assignments(tmp_path, monkeypatch):
    make_inputs(tmp_path, 100)
    monkeypatch.chdir(tmp_path)
    nb = notebook(100)
    ns = {'display': lambda value: None}
    exec(config_source(nb, {'make_plots': False}), ns)
    for identifier in ['runtime-helpers', 'prepare-founding-function', 'h3-functions']:
        exec(source(nb, identifier), ns)
    ns['founding'] = ns['prepare_founding']()
    expected, _, _, _ = ns['prepare_h3_counts']()
    sparse = pd.read_parquet(ns['H3_BIRTHS_PATH'])
    firms = sparse.loc[sparse.index.repeat(sparse.births)].reset_index(drop=True)
    firms['firm_id'] = np.arange(len(firms))
    firms['founding_date'] = pd.PeriodIndex(firms.period, freq='Q').start_time+pd.Timedelta(days=10)
    firms = firms[['firm_id', 'grid_id', 'Fachgruppe_ID', 'founding_date']].rename(columns={'grid_id': 'grid_id_100m'})
    firms.to_parquet(ns['FIRMS_PATH'], index=False)
    ns['H3_BIRTHS_PATH'].unlink()
    actual, keep, _, report = ns['prepare_h3_counts']()
    pd.testing.assert_frame_equal(actual, expected)
    assert keep.all()
    assert report['source'].endswith('firms_assigned_100m.geoparquet')


def test_h3_source_change_invalidates_only_h3_fit_cache(tmp_path, monkeypatch):
    make_inputs(tmp_path, 200)
    first = execute(tmp_path, 200, monkeypatch)
    sparse = pd.read_parquet(first['H3_BIRTHS_PATH'])
    target = sparse.loc[sparse.period.eq('2018Q1') & sparse.Fachgruppe_ID.eq('118')].index[0]
    sparse.loc[target, 'Fachgruppe_ID'] = '201'  # Same total and source schema; group allocation changes.
    sparse = sparse.groupby(['grid_id', 'period', 'Fachgruppe_ID'], as_index=False).births.sum()
    sparse.to_parquet(first['H3_BIRTHS_PATH'], index=False)
    second = execute(tmp_path, 200, monkeypatch, {'run_nb2': False, 'run_cox': False})
    assert second['founding_fit']['loaded_from_cache'] and second['cox_fit']['loaded_from_cache']
    assert second['manifest']['h3']['status'] == 'unavailable'
    result = second['founding_hypotheses'].set_index('hypothesis_id').loc['H3']
    assert result.coverage == 'unavailable'
    assert 'no exactly matching model cache' in result.inference_status
    assert pd.isna(result.p_value)
