"""Behavioral checks for the canonical analysis, compact exports and plotting workflow."""
import ast
import json
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

ROOT=Path(__file__).resolve().parents[1]

def read_notebook(stage):
    name = '08_test_model_assumptions.ipynb' if stage == 8 else '09_estimate_and_save_models.ipynb'
    return json.loads((ROOT/'ANAL'/name).read_text(encoding='utf-8'))

def code(nb, identifier):
    return ''.join(next(c for c in nb['cells'] if c.get('id')==identifier)['source'])

def functions(nb, *identifiers):
    ns=dict(np=np,pd=pd,stats=stats,PHReg=PHReg,perf_counter=perf_counter,CONFIG={'maxiter_cox':30})
    for identifier in identifiers:
        tree=ast.parse(code(nb,identifier))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        exec(compile(tree,identifier,'exec'),ns)
    return ns

def make_base_inputs(root, size):
    """Diverse founding cohorts and tied exits exercise the actual start-stop Cox model."""
    data = root / 'ANAL/data' / f'{size}m'
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


def make_inputs(root, size):
    make_base_inputs(root,size)
    data=root/'ANAL/data'/f'{size}m'
    rng=np.random.default_rng(771)
    for path in (data/'routing/features').glob('*/accessibility_potentials*'):
        f=pd.read_parquet(path)
        f['pop_access_30min']=f.pop_access_15min+rng.gamma(4,300,len(f))
        f['existing_firms_access_30min']=f.existing_firms_access_15min+rng.poisson(20,len(f))
        f.to_parquet(path,index=False)
    for path in (data/'routing/features').glob('*/pedestrian*'):
        f=pd.read_parquet(path)
        f['walk_pt_departures_10min']=f.walk_pt_routes_10min*rng.uniform(2,90,len(f))
        # A physical stop may have no service on the selected reference dates.
        f.loc[f.index[0],['pt_ohne_haltestelle','walk_pt_routes_10min','walk_pt_departures_10min']]=[0,0,0]
        f.to_parquet(path,index=False)
    for path in (data/'routing/features').glob('*/firm_accessibility*'):
        f=pd.read_parquet(path)
        f['existing_firms_access_15min']=f.same_fachgruppe_firms_access_15min+f.own_cell_firms-f.own_cell_same_fachgruppe_firms+rng.poisson(20,len(f))
        f['pop_access_30min']=f.pop_access_15min+rng.gamma(4,300,len(f))
        f['same_fachgruppe_firms_access_30min']=f.same_fachgruppe_firms_access_15min+rng.poisson(5,len(f))
        f['existing_firms_access_30min']=f.existing_firms_access_15min+(f.same_fachgruppe_firms_access_30min-f.same_fachgruppe_firms_access_15min)+rng.poisson(15,len(f))
        f.to_parquet(path,index=False)
    for path in (data/'routing/features').glob('*/nearest*'):
        f=pd.read_parquet(path)
        f['tt_urban_centre_min']=rng.uniform(1,100,len(f))
        f.to_parquet(path,index=False)
    path=data/f'raster_{size}m_styria.geoparquet'
    grid=pd.read_parquet(path)
    grid['centroid_x']=4_000_000+(np.arange(len(grid))%10//2)*6000+(np.arange(len(grid))%2)*size
    grid['centroid_y']=2_600_000+np.arange(len(grid))//10*6000
    grid.to_parquet(path,index=False)
    panel=pd.read_parquet(data/f'raster_quarter_panel_{size}m.parquet')
    rows=panel.loc[panel.year.ge(2016)].loc[lambda f:f.index.repeat(f.births)].reset_index(drop=True)
    path=data/f'firms_assigned_{size}m.geoparquet'
    firms=pd.read_parquet(path)
    names={str(i):f'Software fixture sector {i}' for i in range(1,8)}
    firms['Sparte_Text']=names['1']
    sector=rng.integers(1,8,len(rows)).astype(str)
    extra=pd.DataFrame(dict(firm_id=np.arange(len(rows))+1000,grid_id=rows.grid_id,
        founding_date=pd.PeriodIndex(rows.period,freq='Q').to_timestamp()+pd.Timedelta(days=9),
        exit_date=pd.NaT,exit_observed=False,Sparte_ID=sector,Sparte_Text=pd.Series(sector).map(names)))
    pd.concat([firms,extra],ignore_index=True).to_parquet(path,index=False)
    (root/'FIG').mkdir(exist_ok=True)
    shutil.copytree(ROOT/'FIG/plotting',root/'FIG/plotting',dirs_exist_ok=True)
    helper=root/'FIG/Model results/model_results_visualizations.py'
    helper.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'FIG/Model results/model_results_visualizations.py',helper)
    return data

def prepare(root,size,monkeypatch):
    monkeypatch.chdir(root)
    ns={'display':lambda _:None}
    for c in read_notebook(8)['cells']:
        if c['cell_type']=='code':
            text=''.join(c['source']).replace('CELL_SIZE = 100',f'CELL_SIZE = {size}')
            exec(compile(text,c['id'],'exec'),ns)
    return ns

def estimate(root,size,monkeypatch, fail_stock=False):
    ns=prepare(root,size,monkeypatch)
    for c in read_notebook(9)['cells']:
        if c['cell_type']=='code':
            text=''.join(c['source'])
            if c['id']=='config':
                text=text.replace('START_YEAR, END_YEAR, LAG_YEAR =',
                    f'CONFIG.update(cell_size={size}, ph_sensitivity=False)\nSTART_YEAR, END_YEAR, LAG_YEAR =')
            original = ns.get('count_estimate')
            if fail_stock and c['id']=='count-diagnostics':
                def failed_stock(*args, **kwargs):
                    raise RuntimeError('Injected stock sensitivity failure')
                ns['count_estimate']=failed_stock
            exec(compile(text,c['id'],'exec'),ns)
            if fail_stock and c['id']=='count-diagnostics':
                ns['count_estimate']=original
    return ns


def test_newton_matches_library_likelihood_and_rejects_unfinished_fit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(300, 3))
    stop = np.ceil(rng.exponential(np.exp(-x @ [.4, -.2, .1]))*8)+1
    model = PHReg(stop, x, status=rng.binomial(1, .8, 300), ties='efron')
    ns = functions(read_notebook(9), 'cox-score-helpers')
    fit = ns['cox_newton'](model)
    reference = model.fit(method='newton', maxiter=100, tol=1e-10)
    assert abs(fit['loglike']-reference.llf) < 1e-6
    np.testing.assert_allclose(fit['params'], reference.params, atol=2e-5)
    assert fit['newton_step_se'] <= .005
    with pytest.raises(RuntimeError, match='no convergence'):
        ns['cox_newton'](model, maxiter=0)
    # Warm starts of nested sensitivities must already be on the original coefficient scale.
    warm = ns['cox_newton'](model, start=fit['params'])
    assert warm['iterations'] == 0

def test_existing_efron_cluster_score_validation_still_passes():
    ns = functions(read_notebook(9), 'cox-score-helpers')
    ns['verify_cox_cluster_implementation']()

def test_h5_joint_test_uses_cross_sector_covariance():
    nb = read_notebook(9)
    ns = functions(nb, 'hypothesis-definitions', 'h5-functions')
    estimates = [.2, -.3, .1]
    influence = np.array([[.1, .09, -.02], [-.1, -.08, .03], [.02, -.01, .04], [-.02, 0, -.05]])
    row = ns['h5_heterogeneity'](estimates, list(influence.T), 'cell', True)
    contrast = np.array([[-1, 1, 0], [-1, 0, 1]])
    difference = contrast @ estimates
    variance = contrast @ (influence.T @ influence) @ contrast.T
    assert row['statistic'] == pytest.approx(difference @ np.linalg.solve(variance, difference))
    assert row['df'] == 2
    independent = contrast @ np.diag(np.diag(influence.T @ influence)) @ contrast.T
    assert not np.isclose(row['statistic'], difference @ np.linalg.solve(independent, difference))

def test_all_unidentified_sectors_are_reported_without_breaking_export():
    nb = read_notebook(9)
    ns = functions(nb, 'hypothesis-definitions', 'h5-functions')
    aligned = pd.DataFrame(0, index=range(4), columns=list('1234567'))
    frame = pd.DataFrame({'grid_id': list('abcd'), 'spatial_block': list('abcd')})
    ns.update(founding=frame, X_nb_scaled=pd.DataFrame({'log_tt_motorway_exit': [1., 2., 3., 4.]}),
              nb_scale=pd.Series({'log_tt_motorway_exit': 1.}), display=lambda _: None,
              prepare_h5_counts=lambda: (aligned, np.ones(4, dtype=bool),
                  {s: s for s in '1234567'}, pd.DataFrame(), {}))
    ns.update(record_tests=lambda *a:None, coverage_frames={}, manifest={}, save_results=lambda *a:None)
    text = code(nb, 'h5-analysis')
    exec(text, ns)
    assert ns['h5_diagnostics'].status.eq('unidentified').all()
    assert ns['h5_hypotheses'].coverage.eq('unidentified').all()


def test_active_notebooks_compile_and_do_not_hash_model_inputs_or_refit_figures():
    for folder in ['ANAL','FIG']:
        for path in (ROOT/folder).glob('*.ipynb'):
            nb=json.loads(path.read_text(encoding='utf-8'))
            for c in nb['cells']:
                if c['cell_type']=='code':compile(''.join(c['source']),str(path),'exec')
    for stage in [8,9]:
        text='\n'.join(''.join(c['source']) for c in read_notebook(stage)['cells'])
        assert 'hashlib' not in text and 'file_hash' not in text and 'legacy_100m' not in text
    figure=json.loads((ROOT/'FIG/Model results/10_visualize_model_results.ipynb').read_text(encoding='utf-8'))
    assert all('PHReg' not in ''.join(c['source']) and '.fit(' not in ''.join(c['source']) for c in figure['cells'])

@pytest.mark.parametrize('size',[100,200,500])
def test_canonical_preparation_preserves_zero_counts_and_distinguishes_service(size,tmp_path,monkeypatch):
    make_inputs(tmp_path,size)
    ns=prepare(tmp_path,size,monkeypatch)
    assert ns['DATA_DIR']==tmp_path/'ANAL/data'/f'{size}m'
    for frame in [ns['founding'],ns['survival_frame']]:
        assert np.isfinite(frame.log_walk_pt_departures_10min).all()
    assert (ns['founding'].pt_ohne_haltestelle.eq(0)&ns['founding'].pt_ohne_angebot.eq(1)).any()
    exec(code(read_notebook(9),'h5-functions'),ns)
    aligned,keep,names,coverage,report=ns['prepare_h5_counts']()
    np.testing.assert_array_equal(aligned.sum(axis=1),ns['founding'].births)
    assert keep.all() and set(names)==set('1234567')
    assert aligned[list('1234567')].eq(0).any().all()
    f=pd.read_parquet(ns['FIRMS_PATH'])
    f.loc[f.founding_date.ge('2016-01-01').idxmax(),'Sparte_ID']=None
    f.to_parquet(ns['FIRMS_PATH'],index=False)
    _,_,_,_,report=ns['prepare_h5_counts']()
    assert report['unclassified_births']==1 and report['rows_excluded_unknown']==1

def test_gtfs_routes_require_service_on_the_selected_dates():
    nb=json.loads((ROOT/'OGD/Public_Transport/Public_Transport_preparation.ipynb').read_text(encoding='utf-8'))
    ns=functions(nb,'gtfs-functions')
    chunk=pd.DataFrame(dict(station_id=[1,1,1,2],route_id=['weekday','weekend','holiday','weekend'],
        weekday_school_departures=[1.,0.,0.,0.],weekday_holiday_departures=[0.,0.,.2,0.]))
    active=ns['gtfs_active_station_routes'](chunk)
    assert active.route_id.tolist()==['weekday','holiday']
    assert set(active.station_id)=={1}

def test_wrong_preparation_profile_still_stops_before_fitting(tmp_path,monkeypatch):
    make_inputs(tmp_path,100)
    ns=prepare(tmp_path,100,monkeypatch)
    path=ns['PREPARED_DIR']/'model_data_info.json'
    info=json.loads(path.read_text(encoding='utf-8'))
    info['resolution_m']=500
    path.write_text(json.dumps(info),encoding='utf-8')
    for identifier in ['config','runtime-helpers']:
        exec(code(read_notebook(9),identifier),ns)
    with pytest.raises(ValueError,match='profile mismatch'):
        exec(code(read_notebook(9),'load-prepared'),ns)

def test_optional_stock_failure_does_not_stop_main_models(tmp_path,monkeypatch):
    make_inputs(tmp_path,100)
    ns=estimate(tmp_path,100,monkeypatch,fail_stock=True)
    assert ns['manifest']['status']=='completed'
    assert 'survival_cox' in ns['coefficient_frames'] and 'h5_1' in ns['coefficient_frames']
    assert 'founding_stock_squared' not in ns['coefficient_frames']
    assert any(r['model']=='founding_stock_squared' and r['status']=='FAILED' for r in ns['diagnostic_rows'])

def test_all_seven_outputs_sensitivities_failure_recovery_and_figures(tmp_path,monkeypatch):
    make_inputs(tmp_path,100)
    ns=estimate(tmp_path,100,monkeypatch)
    folder=ns['BUNDLE_DIR']
    expected={'koeffizienten.csv','hypothesentests.csv','diagnostik.csv','stichprobe.csv','plotdaten.parquet','kovarianzen.parquet','laufinfo.json'}
    assert {p.name for p in folder.iterdir()}==expected
    tests=pd.read_csv(folder/'hypothesentests.csv')
    assert {'H4_frequency','H4_joint','H5_heterogeneity'}.issubset(tests.hypothesis_id)
    joint=tests.loc[tests.model.eq('h5_sectors')&tests.hypothesis_id.eq('H5_heterogeneity')&tests.covariance.eq('cell')].iloc[0]
    assert joint.coverage=='tested' and joint.df==6
    assert {'founding_access_30min','founding_centre_control','founding_without_2020','survival_access_30min'}.issubset(tests.model)
    main=ns['coefficient_frames']['survival_cox'].copy()
    state=ns['X_cox']
    ns['CONFIG'].update(ph_sensitivity=True,maxiter_cox_sensitivity=0)
    exec(code(read_notebook(9),'age-sensitivity'),ns)
    assert ns['X_cox'] is state
    assert ns['manifest']['age_sensitivity']['status']=='failed'
    assert ns['manifest']['status']=='completed'
    pd.testing.assert_frame_equal(main,ns['coefficient_frames']['survival_cox'])
    ns['CONFIG']['maxiter_cox_sensitivity']=30
    exec(code(read_notebook(9),'age-sensitivity'),ns)
    assert ns['manifest']['age_sensitivity']['status']=='completed'
    assert 'survival_age_varying' in ns['coefficient_frames']
    # A subsequent failed retry must remove previous sensitivity estimates.
    ns['CONFIG']['maxiter_cox_sensitivity']=0
    exec(code(read_notebook(9),'age-sensitivity'),ns)
    assert 'survival_age_varying' not in ns['coefficient_frames']
    for p in folder.iterdir():
        if p.suffix in ['.csv','.parquet']:
            frame=pd.read_csv(p) if p.suffix=='.csv' else pd.read_parquet(p)
            assert not {'grid_id','firm_id','standort_id'}.intersection(frame.columns)
    cov=pd.read_parquet(folder/'kovarianzen.parquet')
    matrix=cov.loc[cov.model.eq('survival_cox')&cov.covariance.eq('cell')].pivot(index='term_i',columns='term_j',values='value')
    np.testing.assert_allclose(matrix,matrix.T)
    plotting={'display':lambda _:None}
    figure=json.loads((ROOT/'FIG/Model results/10_visualize_model_results.ipynb').read_text(encoding='utf-8'))
    for c in figure['cells']:
        if c['cell_type']=='code':exec(compile(''.join(c['source']),c['id'],'exec'),plotting)
    assert len(list(plotting['OUTPUT_DIR'].glob('*.png')))==12
    assert (plotting['MODEL_FIGURE_DIR']/'figure_kompakter_rastervergleich.svg').is_file()
    plotted=plotting['chapter_data']
    h1=plotted.loc[plotted.figure.eq('R1')&plotted.key.eq('H1')].iloc[0]
    exported=tests.loc[tests.model.eq('founding_nb2')&tests.hypothesis_id.eq('H1')&tests.covariance.eq('cell')].iloc[0]
    np.testing.assert_allclose([h1.estimate,h1.lower,h1.upper],np.exp([exported.estimate,exported.ci_lower,exported.ci_upper]))
    assert '200 m fehlt' in (plotting['MODEL_FIGURE_DIR']/'begleitprotokoll.txt').read_text(encoding='utf-8')
    assert 'H4' in (plotting['OUTPUT_DIR']/'ergebnisuebersicht.md').read_text(encoding='utf-8')
    # The next explicit run preserves the previous results in ARCHIVE.
    exec(code(read_notebook(9),'config'),{'display':lambda _:None})
    archived=list((tmp_path/'ARCHIVE/results/100m').glob('final_*'))
    assert len(archived)==1 and (archived[0]/'koeffizienten.csv').is_file()


def foundation_dates():
    nb=json.loads((ROOT/'ANAL/01_build_data_foundation.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(code(nb,'dc8baeb7'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['parse_event_dates','date_to_quarter']]
    scope=dict(pd=pd,np=np,DATE_DAYFIRST=True)
    exec(compile(tree,'foundation-date-helpers','exec'),scope)
    return nb,scope


def test_chapter_figures_use_exported_cell_contrasts_and_skip_failed_models(tmp_path):
    """A failed Cox fit must not hide valid counts or borrow an older result."""
    import importlib.util
    import matplotlib.pyplot as plt
    module_path=ROOT/'FIG/Model results/model_results_visualizations.py'
    spec=importlib.util.spec_from_file_location('chapter_figures',module_path)
    mrv=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrv)
    models=['founding_nb2','survival_cox','h3_A','h3_C','h5_1']
    terms={'founding_nb2':['log_own_pop','log_tt_motorway_exit','walk_pt_routes_10min',
                          'log_walk_pt_departures_10min','pt_ohne_haltestelle'],
           'survival_cox':['log_own_same','log_own_other'],
           'h3_A':['unused'],'h3_C':['unused'],'h5_1':['log_tt_motorway_exit']}
    coefficients=pd.DataFrame([dict(model=model,term=term,specification='main',
        coefficient=.1,ci_lower=-.1,ci_upper=.3) for model in models for term in terms[model]])
    hypotheses=pd.DataFrame([dict(model=model,hypothesis_id=identifier,specification='main',
        covariance='cell',estimate=estimate,ci_lower=estimate-.12,ci_upper=estimate+.12)
        for model,identifier,estimate in [('founding_nb2','H1',-.4),('founding_nb2','H2a',.3),
            ('founding_nb2','H4_routes',-.002),('founding_nb2','H4_frequency',.01),
            ('founding_nb2','H4_no_stop',-.2),('h3_A_vs_C','H3',-.1),
            ('survival_cox','H2b',.04),('survival_cox','H2c',-.07)]])
    # A conflicting block interval must never enter a chapter figure.
    block=hypotheses.iloc[[0]].assign(covariance='block',estimate=8,ci_lower=7,ci_upper=9)
    hypotheses=pd.concat([hypotheses,block],ignore_index=True)
    diagnostics=pd.DataFrame([dict(model=model,check='fit',status='estimated',
        specification='main',detail='[]',sparte=np.nan,births_retained=np.nan)
        for model in models])
    plotdata=pd.DataFrame([dict(plot_type='h3_population',group=group,population_effect=effect,
        ci_lower=effect-.3,ci_upper=effect+.3,births=count)
        for group,effect,count in [('A',.1,120),('C',.2,80)]])
    covariances=pd.DataFrame([dict(model=model,covariance='cell',term_i='x',term_j='x',value=.01)
        for model in models])
    for size in [100,200,500]:
        folder=tmp_path/'RESULTS'/f'{size}m'/'final'
        folder.mkdir(parents=True)
        info=dict(status='completed' if size!=500 else 'failed',resolution_m=size,run_name='final',
                  created_at='2026-11-06T12:00:00+00:00',pipeline_version='final-specification')
        (folder/'laufinfo.json').write_text(json.dumps(info),encoding='utf-8')
        diag=diagnostics.copy()
        if size==200:
            diag.loc[diag.model.eq('survival_cox'),'status']='FAILED'
        for filename,frame in [('koeffizienten.csv',coefficients),('hypothesentests.csv',hypotheses),
                               ('diagnostik.csv',diag),('stichprobe.csv',pd.DataFrame({'rows':[100]}))]:
            frame.to_csv(folder/filename,index=False)
        plotdata.to_parquet(folder/'plotdaten.parquet',index=False)
        covariances.to_parquet(folder/'kovarianzen.parquet',index=False)
    bundles,notes=mrv.load_bundles(tmp_path,{100:'final',200:'final',500:'final'})
    assert set(bundles)=={100,200} and any('500 m ausgelassen' in note for note in notes)
    data=mrv.prepare_chapter_data(bundles,notes)
    h1=data.loc[data.figure.eq('R1')&data.key.eq('H1')]
    np.testing.assert_allclose(h1.estimate,np.exp(-.4))
    routes=data.loc[data.key.eq('H4_routes')].iloc[0]
    np.testing.assert_allclose([routes.estimate,routes.lower,routes.upper],
                              np.exp(10*np.array([-.002,-.122,.118])))
    difference=data.loc[data.figure.eq('R2')&data.key.eq('H3')].iloc[0]
    np.testing.assert_allclose([difference.lower,difference.upper],np.exp([-.22,.02]))
    cox=data.loc[data.figure.eq('R5')&data.resolution.eq(200)]
    assert cox.estimate.isna().all() and cox.status.eq('nicht schätzbar').all()
    sectors=data.loc[data.figure.eq('R4')&data.role.eq('Punkt')]
    assert sectors.key.tolist()==list('1234567')
    assert sectors.loc[sectors.key.eq('2'),'status'].item()=='nicht schätzbar'
    assert data.loc[data.figure.eq('R4'),'resolution'].eq(100).all()
    # A third successful run adds triangles below the other grids, with the same data sources.
    bundles[500]=dict(bundles[100],run_id='500m/final@fixture')
    data=mrv.prepare_chapter_data(bundles,[])
    fig=mrv.draw_forest(data.loc[data.figure.eq('R1')],'R1',tmp_path)
    plot=fig.axes[1]
    markers={line.get_marker() for line in plot.lines}
    assert {'o','s','^'}.issubset(markers)
    np.testing.assert_allclose(fig.get_size_inches()[0]*2.54,16)
    assert plot.get_xscale()=='log'
    csv=pd.read_csv(tmp_path/(mrv.FIGURE_NAMES['R1']+'.csv'))
    assert set(csv.Rasterweite)=={100,200,500}
    assert not csv['Lauf-ID'].isna().any()
    plt.close(fig)


def test_event_dates_accept_mixed_complete_dates_and_preserve_real_blanks():
    _,scope=foundation_dates()
    values=pd.Series(['31.12.2025','2025-12-31 12:00','04/05/2020',None,'  ',pd.Timestamp('2015-01-01')])
    actual=scope['parse_event_dates'](values,'source_date')
    assert actual.iloc[0]==pd.Timestamp('2025-12-31')
    assert actual.iloc[1]==pd.Timestamp('2025-12-31 12:00')
    assert actual.iloc[2]==pd.Timestamp('2020-05-04')
    assert actual.iloc[3:5].isna().all()
    assert actual.iloc[5]==pd.Timestamp('2015-01-01')


@pytest.mark.parametrize('value',['not-a-date','30.02.2020','2020-13-10','10.13.2020',44205,20191231,'2020','2020-12','May 2020','NaT'])
def test_nonempty_bad_dates_cannot_become_censored_exits(value):
    _,scope=foundation_dates()
    with pytest.raises(ValueError,match='invalid dates'):
        scope['parse_event_dates'](pd.Series([None,value]),'exit_date')


@pytest.mark.parametrize('dayfirst',[True,False])
def test_year_first_dates_never_swap_month_and_day(dayfirst):
    _,scope=foundation_dates()
    scope['DATE_DAYFIRST']=dayfirst
    source=pd.Series(['2020-05-04','2020-05-04 12:00','2020/05/04','20200504'])
    actual=scope['parse_event_dates'](source,'founding_date')
    assert actual.dt.month.tolist()==[5]*4
    assert actual.dt.day.tolist()==[4]*4
    assert actual.dt.hour.tolist()==[0,12,0,0]


def test_analysis_end_includes_the_last_day_but_not_the_next_year():
    nb,_=foundation_dates()
    config=ast.parse(code(nb,'e4b8a812'))
    assignment=next(n for n in config.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CENSORING_DATE' for t in n.targets))
    scope={'pd':pd,'END_YEAR':2025}
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'analysis-end','exec'),scope)
    assert pd.Timestamp('2025-12-31 23:59:59')<=scope['CENSORING_DATE']<pd.Timestamp('2026-01-01')


def test_reversed_dates_fail_before_assignment_is_written():
    nb,scope=foundation_dates()
    scope.update(firms_source=pd.DataFrame({'born':['2020-01-02'],'died':['2020-01-01']}),FOUNDING_DATE_COLUMN='born',EXIT_DATE_COLUMN='died')
    preparation=code(nb,'09916915').split('# Assign firms')[0]
    with pytest.raises(ValueError,match='Exit date before founding date'):
        exec(preparation,scope)


def test_foundation_keeps_last_day_events_and_excludes_exited_firms_from_later_stocks(tmp_path):
    nb,scope=foundation_dates()
    raster=pd.DataFrame({'grid_id':['g1','g2','g3'],'municipality_id':['m']*3,'municipality_name':['M']*3,
        'population_2019':[1.,1.,0.],'population_2025':[1.,1.,0.]})
    firms=pd.DataFrame({'firm_id':[1,2,3],'grid_id':['g1','g2','g3'],'Fachgruppe_ID':['101','102','101'],
        'founding_date':pd.to_datetime(['2014-01-01','2015-12-31 12:00','2015-01-01'],format='mixed'),
        'exit_date':pd.to_datetime(['2015-06-30 12:00',None,'2015-01-01'],format='mixed')})
    firms['founding_quarter']=firms.founding_date.dt.to_period('Q').astype('string')
    firms['exit_quarter']=firms.exit_date.dt.to_period('Q').astype('string')
    pop=pd.MultiIndex.from_product([['g1','g2','g3'],[2015],[1,2,3,4]],names=['grid_id','year','quarter']).to_frame(index=False)
    pop['population_backcast']=pop.grid_id.map({'g1':1.,'g2':1.,'g3':0.})
    scope.update(raster=raster,firms=firms,population_backcast=pop,START_YEAR=2015,END_YEAR=2015,
        CENSORING_DATE=pd.Period('2015Q4',freq='Q').end_time,FACHGRUPPE_IDS=['101','102'],
        POPULATION_BACKCAST_OUTPUT=tmp_path/'population.parquet',PANEL_OUTPUT=tmp_path/'panel.parquet',
        BIRTHS_BY_FACHGRUPPE_OUTPUT=tmp_path/'births.parquet')
    exec(code(nb,'9f1ee32d'),scope)
    panel=scope['panel'].set_index(['grid_id','quarter'])
    assert panel.loc[('g2',4),'births']==1
    assert panel.loc[('g2',4),'active_firms_t']==1
    assert panel.loc[('g1',2),'active_firms_t']==0
    assert panel.loc[('g1',3),'active_firms_tminus1']==0
    assert panel.loc[('g3',1),'births']==panel.loc[('g3',1),'exits']==1


def regional_notebook():
    return json.loads((ROOT/'ANAL/09b_estimate_graz_vs_rest.ipynb').read_text(encoding='utf-8'))


def regional_functions():
    nb = regional_notebook()
    namespace = {'display': lambda _: None}
    for identifier in ['config', 'reuse-functions', 'region-helpers', 'regional-count',
                       'regional-cox', 'regional-comparisons']:
        exec(compile(code(nb, identifier), identifier, 'exec'), namespace)
    return namespace


def test_regional_assignment_preserves_rows_and_whole_location_histories():
    ns = regional_functions()
    municipality_map = ns['municipality_ids'](pd.Series([60101., 60608.], index=['g1', 'g2']))
    frame = pd.DataFrame({'standort_id': [1, 1, 2], 'grid_id': ['g1', 'g1', 'g2'],
                          'period': ['2016Q1', '2016Q2', '2016Q1'], 'event': [0, 1, 0]})
    result = ns['assign_regions'](frame, municipality_map)
    pd.testing.assert_frame_equal(result.drop(columns='region'), frame)
    assert result.region.tolist() == ['graz', 'graz', 'rest_styria']
    assert result.groupby('standort_id').region.nunique().eq(1).all()
    with pytest.raises(ValueError, match='beide Regionen'):
        ns['assign_regions'](frame.assign(standort_id=[1, 1, 1]), municipality_map)
    with pytest.raises(ValueError, match='ohne eindeutige Gemeindezuordnung'):
        ns['assign_regions'](frame.assign(grid_id=['unknown', 'g1', 'g2']), municipality_map)
    with pytest.raises(ValueError, match='widerspricht'):
        ns['assign_regions'](frame.assign(municipality_id='60608'), municipality_map)


def test_regional_differences_use_full_h1_covariance_and_joint_h4_covariance():
    ns = regional_functions()
    names = ['log_pop_access_ring_0_15', 'log_firms_relative_car_ring_0_15',
             *ns['H4_TERMS'], 'log_tt_motorway_exit']
    covariance_graz = np.eye(len(names)) * .04
    covariance_graz[0, 1] = covariance_graz[1, 0] = .015
    covariance_rest = np.eye(len(names)) * .09
    covariance_rest[2, 3] = covariance_rest[3, 2] = .02
    params_graz = np.array([.4, .1, .2, -.1, .3, -.2])
    params_rest = np.array([.1, .2, -.1, .2, .1, -.1])
    # Different column order exercises matching by term name.
    order = np.arange(len(names))[::-1]
    fits = {
        'founding_graz': dict(names=names, params=params_graz, covariance=covariance_graz, family='poisson'),
        'founding_rest_styria': dict(names=np.array(names)[order].tolist(), params=params_rest[order],
                                   covariance=covariance_rest[np.ix_(order, order)], family='nb2'),
    }
    comparisons = ns['regional_comparisons'](fits).set_index('hypothesis_id')
    h1 = comparisons.loc['H1']
    weights = np.array([1, -1, 0, 0, 0, 0])
    expected_variance = weights @ (covariance_graz + covariance_rest) @ weights
    assert h1.estimate == pytest.approx(.4)
    assert h1.std_error == pytest.approx(np.sqrt(expected_variance))
    assert h1.ci_lower == pytest.approx(.4 - 1.96 * np.sqrt(expected_variance))
    assert h1.effect_ratio == pytest.approx(np.exp(.4))
    assert h1.p_value == pytest.approx(2 * stats.norm.sf(abs(.4 / np.sqrt(expected_variance))))
    assert h1.family_graz == 'poisson'
    difference = params_graz[2:5] - params_rest[2:5]
    joint_covariance = (covariance_graz + covariance_rest)[2:5, 2:5]
    assert comparisons.loc['H4_joint', 'statistic'] == pytest.approx(
        difference @ np.linalg.solve(joint_covariance, difference))
    assert comparisons.loc['H4_joint', 'df'] == 3
    assert comparisons.loc['H2b', 'coverage'] == 'unavailable'
    unavailable = ns['regional_comparisons']({})
    assert unavailable.coverage.eq('unavailable').all()
    assert unavailable[['estimate', 'std_error', 'ci_lower', 'ci_upper', 'p_value']].isna().all().all()


def test_regional_constant_terms_and_graz_poisson_fallback_are_explicit():
    ns = regional_functions()
    rng = np.random.default_rng(91)
    frame = pd.DataFrame(rng.normal(size=(160, len(ns['FOUNDING_TERMS']))),
                         columns=ns['FOUNDING_TERMS'])
    frame['grid_id'] = np.repeat(np.arange(40), 4)
    frame['period'] = np.tile(['2016Q1', '2016Q2', '2016Q3', '2016Q4'], 40)
    frame['births'] = rng.poisson(2, len(frame))
    frame['pt_ohne_haltestelle'] = 0
    calls = []

    def fail_nb2(family, X, y, groups, scale, poisson=None):
        calls.append((family, list(X.columns)))
        if family == 'nb2':
            raise RuntimeError('Injected NB2 nonconvergence')
        return dict(family='poisson', names=list(X.columns), params=np.zeros(X.shape[1]),
                    covariance=np.eye(X.shape[1]))

    ns['count_estimate'] = fail_nb2
    fit = ns['estimate_regional_count'](frame, 'graz')
    assert fit['family'] == 'poisson'
    assert fit['fallback_reason'] == 'Injected NB2 nonconvergence'
    assert fit['excluded_terms'] == ['population_growth_yoy']
    assert 'population_growth_yoy' not in calls[0][1]
    assert fit['constant_terms'] == ['pt_ohne_haltestelle']
    tests = ns['hypothesis_table'](fit, ns['COUNT_TESTS']).set_index('hypothesis_id')
    assert tests.loc['H4_no_stop', 'coverage'] == 'unidentified'
    comparisons = ns['regional_comparisons']({'founding_graz': fit, 'founding_rest_styria': fit})
    assert comparisons.loc[comparisons.hypothesis_id.isin(['H4_no_stop', 'H4_joint']),
                           'coverage'].eq('unidentified').all()
    with pytest.raises(RuntimeError, match='Injected NB2 nonconvergence'):
        ns['estimate_regional_count'](frame, 'rest_styria')
    assert 'population_growth_yoy' in calls[-1][1]


@pytest.mark.parametrize('failed_region', [None, 'graz', 'rest_styria'])
def test_regional_notebook_exports_four_models_and_continues_after_failure(tmp_path, monkeypatch, failed_region):
    data = make_inputs(tmp_path, 100)
    grid_path = data/'raster_100m_styria.geoparquet'
    grid = pd.read_parquet(grid_path)
    # Each region contains different founding cohorts within every Fachgruppe.
    grid['municipality_id'] = np.where(np.arange(len(grid)) < len(grid) // 2, '60101', '60608')
    grid.to_parquet(grid_path, index=False)
    municipality_map = grid.set_index('grid_id').municipality_id
    panel_path = data/'raster_quarter_panel_100m.parquet'
    panel = pd.read_parquet(panel_path)
    panel['municipality_id'] = panel.grid_id.map(municipality_map)
    panel.to_parquet(panel_path, index=False)
    prepare(tmp_path, 100, monkeypatch)
    # Municipality IDs suffice for 9b, even with no projected coordinates available.
    grid[['grid_id', 'municipality_id']].to_parquet(grid_path, index=False)
    shutil.copy2(ROOT/'ANAL/09_estimate_and_save_models.ipynb', tmp_path/'ANAL')
    main_folder = tmp_path/'RESULTS/100m/final'
    main_folder.mkdir(parents=True)
    (main_folder/'existing_result.csv').write_bytes(b'preserved main result\n')
    namespace = {'display': lambda _: None}
    nb = regional_notebook()
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
        if cell['id'] == 'reuse-functions' and failed_region is not None:
            original_count_estimate = namespace['count_estimate']
            failed_code = '60101' if failed_region == 'graz' else '60608'
            failed_cells = set(grid.loc[grid.municipality_id.eq(failed_code), 'grid_id'])

            def fail_regional_nb2(family, X, y, groups, scale, poisson=None, start=None):
                if family == 'nb2' and set(groups).issubset(failed_cells):
                    raise RuntimeError('Injected regional NB2 failure')
                return original_count_estimate(family, X, y, groups, scale, poisson=poisson, start=start)

            namespace['count_estimate'] = fail_regional_nb2
    folder = tmp_path/'RESULTS/graz_vs_rest/100m/final'
    expected = {'koeffizienten.csv', 'hypothesentests.csv', 'diagnostik.csv', 'stichprobe.csv',
                'plotdaten.parquet', 'kovarianzen.parquet', 'laufinfo.json', 'regionalvergleiche.csv'}
    assert {path.name for path in folder.iterdir()} == expected
    manifest = json.loads((folder/'laufinfo.json').read_text(encoding='utf-8'))
    assert set(manifest['outputs']) == expected
    assert (main_folder/'existing_result.csv').read_bytes() == b'preserved main result\n'
    assert set(namespace['regional_fits']) >= {'survival_graz', 'survival_rest_styria'}
    assert pd.read_parquet(folder/'kovarianzen.parquet').covariance.eq('cell').all()
    samples = pd.read_csv(folder/'stichprobe.csv')
    regional_samples = samples.loc[samples.region.isin(['graz', 'rest_styria'])]
    for model_type in ['founding', 'survival']:
        selected = regional_samples.loc[regional_samples.model.str.startswith(model_type + '_')]
        assert selected.rows.sum() == namespace['prepared']['tables'][model_type]['rows']
        assert selected.events.sum() == sum(
            row['events'] for row in namespace['prepared']['sample_description']
            if row['model'] == model_type and row['stage'] == 'model_sample')
    assert 'population_growth_yoy' not in namespace['regional_fits']['founding_graz']['names']
    comparisons = pd.read_csv(folder/'regionalvergleiche.csv')
    assert comparisons.loc[comparisons.model_type.eq('survival'), 'coverage'].eq('tested').all()
    if failed_region == 'rest_styria':
        assert manifest['status'] == 'partially_completed'
        assert manifest['models']['founding_rest_styria']['status'] == 'failed'
        assert comparisons.loc[comparisons.model_type.eq('founding'), 'coverage'].eq('unavailable').all()
    else:
        assert manifest['status'] == 'completed'
        assert len(namespace['regional_fits']) == 4
        assert 'population_growth_yoy' in namespace['regional_fits']['founding_rest_styria']['names']
        if failed_region == 'graz':
            assert manifest['models']['founding_graz']['family'] == 'poisson'
            assert manifest['models']['founding_graz']['fallback_reason'] == 'Injected regional NB2 failure'
    if failed_region is None:
        previous_coefficients = (folder/'koeffizienten.csv').read_bytes()
        exec(code(nb, 'initialize-results'), namespace)
        archived = list((tmp_path/'ARCHIVE/results/graz_vs_rest/100m').glob('final_*'))
        assert len(archived) == 1
        assert (archived[0]/'koeffizienten.csv').read_bytes() == previous_coefficients
        assert folder.is_dir() and not any(folder.iterdir())
