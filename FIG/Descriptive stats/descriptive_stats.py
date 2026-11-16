"""Small reporting helpers for the saved main-model samples; no model fitting."""

import json
import textwrap
from html import escape
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from IPython.display import HTML

from FIG.plotting.apa7 import apa7_axes, set_apa7_style


MODELS = {"founding": "Gründung", "survival": "Überleben"}
MODEL_IDS = {"founding": "founding_nb2", "survival": "survival_cox"}
LABELS = {
    "log_own_pop": "Eigenbevölkerung (log)", "log_own_firms": "Eigenbestand (log)",
    "log_pop_access_ring_0_15": "Bevölkerung Auto (log)",
    "log_firms_relative_car_ring_0_15": "Relativer Bestand Auto",
    "log_walk_pop_ring_0_10": "Bevölkerung Gehen (log)",
    "log_firms_relative_walk_ring_0_10": "Relativer Bestand Gehen",
    "population_growth_yoy": "Bevölkerungsänderung (log)",
    "walk_pt_routes_10min": "ÖV-Linienzahl", "log_walk_pt_departures_10min": "ÖV-Abfahrten (log)",
    "pt_ohne_haltestelle": "Keine Haltestelle", "log_tt_motorway_exit": "Autobahnfahrzeit (log)",
    "log_own_same": "Eigenbestand gleich (log)", "log_own_other": "Eigenbestand fremd (log)",
    "log_pop_ring_0_15": "Bevölkerung Auto (log)",
    "log_same_relative_ring_0_15": "Relativer Bestand gleich",
    "log_other_relative_ring_0_15": "Relativer Bestand fremd",
    "log_absolute_firms_car": "Absoluter Bestand Auto (log)",
}
ORDER = {
    "founding": ["log_own_pop", "log_own_firms", "log_pop_access_ring_0_15",
                 "log_firms_relative_car_ring_0_15", "log_walk_pop_ring_0_10",
                 "log_firms_relative_walk_ring_0_10", "population_growth_yoy",
                 "walk_pt_routes_10min", "log_walk_pt_departures_10min",
                 "pt_ohne_haltestelle", "log_tt_motorway_exit"],
    "survival": ["log_own_pop", "log_own_same", "log_own_other", "log_pop_ring_0_15",
                 "log_same_relative_ring_0_15", "log_other_relative_ring_0_15",
                 "walk_pt_routes_10min", "log_walk_pt_departures_10min",
                 "pt_ohne_haltestelle", "log_tt_motorway_exit"],
}
CENTRES = {"km_urban_centre": "Urban\nkm", "km_regional_centre": "Regional\nkm",
           "tt_urban_centre_min": "Urban\nMinuten", "tt_regional_centre_min": "Regional\nMinuten"}


def german(value, digits=1):
    if pd.isna(value):
        return "–"
    if np.isinf(value):
        return "∞"
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def table(frame, digits=1, caption=""):
    view = frame.copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: german(value, digits))
    heading = f"<b>{escape(caption)}</b>" if caption else ""
    return HTML('<div style="font-family:Arial;font-size:10pt">' + heading +
                view.to_html(index=False, border=0, na_rep="–") + '</div>')


def save(fig, folder, name):
    for extension in ["png", "svg"]:
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(Path(folder) / f"{name}.{extension}", dpi=300, facecolor="white")


def style():
    font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    set_apa7_style()
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"]})


def original_variables(model):
    """Invert the actual model transformations, including the relative stocks."""
    def mass(term):
        return f"greatest(0,exp({term})-1)"

    pop = "log_pop_access_ring_0_15" if model == "founding" else "log_pop_ring_0_15"
    rows = [("Bevölkerung der eigenen Zelle" + (" (Vorjahr)" if model == "founding" else ""),
             "Personen", mass("log_own_pop"), "", "")]
    if model == "founding":
        rows.append(("Unternehmensbestand der eigenen Zelle", "Standorte", mass("log_own_firms"), "", ""))
    else:
        rows += [(f"{name} Bestand der eigenen Zelle", "Standorte", mass(term), "", "")
                 for name, term in [("Branchengleicher", "log_own_same"), ("Branchenfremder", "log_own_other")]]
    rows.append(("Bevölkerung im Autoumfeld (15 Minuten)", "Personen", mass(pop), "", ""))
    stocks = [("Unternehmensbestand im Autoumfeld", "log_firms_relative_car_ring_0_15", pop)] if model == "founding" else [
        ("Branchengleicher Bestand im Autoumfeld", "log_same_relative_ring_0_15", pop),
        ("Branchenfremder Bestand im Autoumfeld", "log_other_relative_ring_0_15", pop)]
    if model == "founding":
        stocks += [("Unternehmensbestand im Gehumfeld", "log_firms_relative_walk_ring_0_10", "log_walk_pop_ring_0_10")]
    for name, relative, population in stocks:
        if "Gehumfeld" in name:
            rows.append(("Bevölkerung im Gehumfeld (10 Minuten)", "Personen", mass(population), "", ""))
        firms = mass(f"{relative}+{population}")
        people = mass(population)
        rows.append((name, "Standorte", firms, "", ""))
        rows.append((name + " je 1.000 Personen", "Standorte / 1.000 Personen",
                     f"CASE WHEN {people}>0 THEN 1000*({firms})/({people}) END", "Dichte", ""))
    if model == "founding":
        rows.append(("Bevölkerungsänderung zum Vorjahr", "Personen",
                     "exp(log_own_pop)*(exp(population_growth_yoy)-1)", "", ""))
    rows += [("Linienzahl in 10 Gehminuten", "Linien", "walk_pt_routes_10min", "", ""),
             ("Abfahrten in 10 Gehminuten", "Abfahrten / Werktag", mass("log_walk_pt_departures_10min"), "", ""),
             ("Keine Haltestelle (Anteil mit Wert 1)", "%", "100.0*pt_ohne_haltestelle", "Indikator", ""),
             ("Fahrzeit zur nächsten Autobahnauffahrt", "Minuten", mass("log_tt_motorway_exit"), "", "")]
    return rows


def descriptive_table(connection, model):
    recipes = original_variables(model)
    expressions = ",".join(f"({expression}) AS x{i}" for i, (_, _, expression, _, _) in enumerate(recipes))
    connection.execute(f"CREATE OR REPLACE TEMP VIEW original AS SELECT {expressions} FROM model")
    queries = []
    for i in range(len(recipes)):
        queries += [f"{function}(x{i}) AS x{i}_{key}" for key, function in
                    [("mean", "avg"), ("median", "median"), ("sd", "stddev_samp"),
                     ("min", "min"), ("max", "max"), ("n", "count")]]
        queries.append(f"100.0*avg(CASE WHEN x{i} IS NULL THEN NULL WHEN abs(x{i})<1e-9 THEN 1 ELSE 0 END) AS x{i}_zero")
    values = connection.execute("SELECT " + ",".join(queries) + " FROM original").df().iloc[0]
    total = connection.execute("SELECT count(*) FROM model").fetchone()[0]
    rows = []
    for i, (name, unit, _, kind, _) in enumerate(recipes):
        row = dict(Variable=name, Einheit=unit, Mittelwert=values[f"x{i}_mean"],
                   Median=values[f"x{i}_median"], SD=values[f"x{i}_sd"],
                   Min=values[f"x{i}_min"], Max=values[f"x{i}_max"],
                   **{"Anteil Null (%)": values[f"x{i}_zero"]},
                   n=int(values[f"x{i}_n"]), **{"Ausgeschlossen (P=0)": total-int(values[f"x{i}_n"]) if kind=="Dichte" else 0})
        if kind == "Indikator":
            for key in ["Median", "SD", "Min", "Max", "Anteil Null (%)"]:
                row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_matrix(connection, model):
    terms = ORDER[model].copy()
    if model == "founding":
        connection.execute("CREATE OR REPLACE TEMP VIEW predictors AS SELECT *, log_pop_access_ring_0_15+log_firms_relative_car_ring_0_15 AS log_absolute_firms_car FROM model")
        terms.append("log_absolute_firms_car")
    else:
        connection.execute("CREATE OR REPLACE TEMP VIEW predictors AS SELECT * FROM model")
    pairs = [(i, j) for i in range(len(terms)) for j in range(i)]
    query = ",".join(f"corr({terms[i]},{terms[j]}) AS c{i}_{j}" for i, j in pairs)
    values = connection.execute("SELECT " + query + " FROM predictors").fetchone()
    matrix = pd.DataFrame(np.eye(len(terms)), index=terms, columns=terms)
    for (i, j), value in zip(pairs, values):
        matrix.iloc[i, j] = matrix.iloc[j, i] = np.nan if value is None else value
    return matrix


def vif_from_correlation(matrix):
    """VIF on the full saved sample, with an intercept and no time controls."""
    result = {}
    for term in matrix:
        others = [c for c in matrix if c != term]
        vector = matrix.loc[others, term].to_numpy()
        correlation = matrix.loc[others, others].to_numpy()
        if not np.isfinite(vector).all() or not np.isfinite(correlation).all():
            result[term] = np.nan
            continue
        remaining = 1 - vector @ np.linalg.pinv(correlation, hermitian=True) @ vector
        result[term] = np.inf if remaining < 1e-10 else 1 / remaining
    return pd.Series(result, name="VIF")


def centre_correlations(connection, files, model):
    connection.from_parquet([str(p) for p in files]).create_view("nearest_raw")
    if connection.execute("SELECT count(*) FROM (SELECT grid_id,year FROM nearest_raw GROUP BY 1,2 HAVING count(*)>1)").fetchone()[0]:
        raise ValueError("Doppelte Zelle-Jahr-Schlüssel in nearest_infrastructure.")
    fields = []
    for column in CENTRES:
        kind = "urban_centre" if "urban" in column else "regional_centre"
        fields.append(f"CASE WHEN routing_status_{kind}='ok' AND isfinite({column}) AND {column}>=0 THEN {column} END AS {column}")
    connection.execute("CREATE OR REPLACE TEMP VIEW nearest AS SELECT grid_id,year," + ",".join(fields) + " FROM nearest_raw")
    queries = [f"corr(m.{term},n.{centre})" for term in ORDER[model] for centre in CENTRES]
    queries += [f"count(n.{centre})" for centre in CENTRES]
    values = connection.execute("SELECT " + ",".join(queries) + " FROM model m LEFT JOIN nearest n USING(grid_id,year)").fetchone()
    matrix = pd.DataFrame(np.array(values[:-4], dtype=float).reshape(-1, 4), index=ORDER[model], columns=CENTRES)
    counts = pd.Series(values[-4:], index=CENTRES, name="n")
    return matrix, counts


def analyse(project, size, run, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = Path(project) / "RESULTS" / f"{size}m" / run
    info = json.loads((bundle / "laufinfo.json").read_text(encoding="utf-8"))
    if info["status"] != "completed" or info["resolution_m"] != size or info["run_name"] != run:
        raise ValueError("Kein passender abgeschlossener Ergebnislauf.")
    diagnostics = pd.read_csv(bundle / "diagnostik.csv")
    sample = pd.read_csv(bundle / "stichprobe.csv")
    sample = sample.loc[sample.model.isin(MODELS) & sample.stage.eq("model_sample")].copy()
    result = dict(sample=sample, stats={}, correlations={}, centres={}, vif=pd.DataFrame(), missing=[])
    report = [f"{size} m / {run} / {info['created_at']}",
              "Exakte Modellstichproben; kein Aggregieren oder zusätzliches Filtern.",
              "Pearson: Modelltransformationen, ohne Zeitkontrollen; keine p-Werte.",
              "Umfeldbestände ohne eigene Zelle. Absolute Werte aus den gespeicherten Log-Variablen rückgerechnet.",
              "Nullanteil: |Wert|<1e-9 als numerische Null; Dichten nur bei P>0."]
    data = Path(project) / "ANAL/data" / f"{size}m"
    meta_path = data / "modelldaten/model_data_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    if not meta:
        result["missing"].append(str(meta_path.relative_to(project)))
    if meta and (meta.get("status"), meta.get("version"), meta.get("resolution_m")) != ("completed", "model-tables-3", size):
        raise ValueError("Modelltabellen passen nicht zum ausgewählten Profil.")
    years = range(info["study_period"]["start_year"], info["study_period"]["end_year"] + 1)
    nearest = [data / "routing/features" / str(y) / f"nearest_infrastructure_{size}m.parquet" for y in years]
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit='2GB'")
        for model in MODELS:
            row = sample.loc[sample.model.eq(model)].iloc[0]
            report.append(f"{MODELS[model]}: {german(row.rows,0)} Beobachtungen; {german(row.units,0)} {'Rasterzellen' if model=='founding' else 'Standorte'}.")
            path = data / "modelldaten" / f"{model}_frame.parquet"
            if not meta or not path.is_file():
                if not path.is_file():
                    result["missing"].append(str(path.relative_to(project)))
                continue
            required = ORDER[model]
            if set(meta[model+"_terms"]) - {"calendar_year"} != set(required):
                raise ValueError(f"{model}: andere Modellvariablen als erwartet.")
            if meta["tables"][model]["file"] != path.name:
                raise ValueError(f"{model}: unerwarteter Tabellenname.")
            connection.from_parquet(str(path)).create_view("model")
            n = connection.execute("SELECT count(*) FROM model").fetchone()[0]
            if n != int(row.rows) or n != meta["tables"][model]["rows"]:
                raise ValueError(f"{model}: Zeilenzahl passt nicht zum Ergebnislauf.")
            unit = "grid_id" if model == "founding" else "standort_id"
            if connection.execute(f"SELECT count(DISTINCT {unit}) FROM model").fetchone()[0] != int(row.units):
                raise ValueError(f"{model}: Untersuchungseinheiten passen nicht zum Ergebnislauf.")
            bad = " OR ".join(f"NOT coalesce(isfinite({term}),false)" for term in required)
            if connection.execute(f"SELECT count(*) FROM model WHERE {bad}").fetchone()[0]:
                raise ValueError(f"{model}: fehlende Modellwerte; keine zusätzlichen Filter erlaubt.")
            result["stats"][model] = descriptive_table(connection, model)
            result["correlations"][model] = correlation_matrix(connection, model)
            if all(p.is_file() for p in nearest):
                result["centres"][model], counts = centre_correlations(connection, nearest, model)
                counts.to_csv(output / f"zentren_n_{model}.csv", sep=";", decimal=",", encoding="utf-8-sig")
                report.append(f"{MODELS[model]} – Zentrenkorrelationen, paarweise n: " + "; ".join(f"{k}={int(v)}" for k,v in counts.items()))
    vif_rows = []
    for model in MODELS:
        source = "Export; mit Quartalskontrollen" if model == "founding" else "Export; mit Kalendertrend"
        values = diagnostics.loc[diagnostics.model.eq(MODEL_IDS[model]) & diagnostics.check.eq("VIF")].set_index("term").value
        if model == "survival" and model in result["correlations"]:
            values = vif_from_correlation(result["correlations"][model])
            source = "Neu; ohne Kalendertrend"
        for term in ORDER[model]:
            vif_rows.append(dict(Modell=MODELS[model], Variable=LABELS[term], Term=term,
                                 VIF=values.get(term, np.nan), Quelle=source))
    result["vif"] = pd.DataFrame(vif_rows)
    for model, table in result["stats"].items():
        table.to_csv(output / f"deskriptiv_{model}.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
        for row in table.loc[table.Variable.str.contains("je 1.000 Personen",regex=False)].itertuples(index=False, name=None):
            report.append(f"{MODELS[model]} – {row[0]}: {row[-1]} Beobachtungen mit P=0 ausgeschlossen (nur Dichtezeile).")
    for model, matrix in result["correlations"].items():
        lower_triangle(matrix).rename(index=LABELS,columns=LABELS).to_csv(output / f"korrelationen_{model}.csv", sep=";", decimal=",", encoding="utf-8-sig")
        pairs = [(matrix.index[i], matrix.columns[j], matrix.iloc[i,j]) for i in range(len(matrix)) for j in range(i) if pd.notna(matrix.iloc[i,j])]
        report += [f"{MODELS[model]} – {LABELS[a]} / {LABELS[b]}: r={german(v,2)}" for a,b,v in sorted(pairs,key=lambda p:abs(p[2]),reverse=True)[:5]]
        if model == "founding":
            pop = "log_pop_access_ring_0_15"
            for term in ["log_absolute_firms_car", "log_firms_relative_car_ring_0_15"]:
                report.append(f"Bevölkerung Auto / {LABELS[term]}: r={german(matrix.loc[pop,term],2)}")
    for model, matrix in result["centres"].items():
        matrix.rename(index=LABELS).to_csv(output / f"zentren_korrelationen_{model}.csv", sep=";", decimal=",", encoding="utf-8-sig")
    for name, frame in [("stichprobe", sample), ("vif", result["vif"])]:
        frame.to_csv(output / f"{name}.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    for model, group in result["vif"].groupby("Modell", sort=False):
        good = group.dropna(subset=["VIF"])
        if not good.empty:
            row = good.loc[good.VIF.idxmax()]
            report.append(f"{model} – höchster VIF: {row.Variable} = {german(row.VIF,2)} ({row.Quelle}).")
    if result["missing"]:
        report += ["Nicht berechnet: Deskriptivwerte/Korrelationen ohne gespeicherte Modelltabellen.", *result["missing"]]
    if not result["centres"]:
        report.append("Zentrenvergleich ausstehend: Modelltabellen und jährliche nearest_infrastructure-Dateien erforderlich.")
    report.append("Zentren: Gemeinde-Repräsentativpunkte; km der nach Fahrzeit gewählten Route. Zentrenkorrelationen paarweise, ohne Veränderung der Modellstichprobe.")
    (output / "kurzprotokoll.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    return result


def lower_triangle(matrix):
    return matrix.where(np.tril(np.ones(matrix.shape, dtype=bool), k=-1))


def heatmap(matrix, folder, name, triangular=False):
    style()
    n, columns = matrix.shape
    height = max(7.5, n * .85 + 3.3)
    fig = plt.figure(figsize=(16/2.54, height/2.54))
    ax = fig.add_axes([.34, 2.4/height, .63, (height-3.1)/height])
    values = lower_triangle(matrix).to_numpy() if triangular else matrix.to_numpy()
    palette = LinearSegmentedColormap.from_list("correlations", ["#245477", "#f2f2f2", "#af4b4b"]).with_extremes(bad="white")
    image = ax.imshow(values, cmap=palette, vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(n), [textwrap.fill((f"{i+1}  " if triangular else "") + LABELS[t], 27) for i,t in enumerate(matrix.index)], fontsize=9)
    ax.set_xticks(range(columns), [str(i+1) for i in range(columns)] if triangular else [CENTRES[t] for t in matrix.columns], fontsize=9)
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i,j), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(j,i,german(value,2),ha="center",va="center",fontsize=9,color="white" if abs(value)>.65 else "black")
        elif not triangular:
            ax.text(j,i,"–",ha="center",va="center",fontsize=9,color=".5")
    cax = fig.add_axes([.46, 1.1/height, .42, .2/height])
    bar = fig.colorbar(image,cax=cax,orientation="horizontal",ticks=[-1,-.5,0,.5,1])
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:german(x,1)))
    bar.set_label("Pearson r",fontsize=9)
    save(fig,folder,name)
    return fig


def vif_plot(table, folder):
    style()
    labels = list(dict.fromkeys(table.Variable))
    height = len(labels) * .7 + 2.7
    fig = plt.figure(figsize=(16/2.54,height/2.54))
    ax = fig.add_axes([.44,1.6/height,.53,(height-2.7)/height])
    for model, offset, colour in [("Gründung",-.16,"#245477"),("Überleben",.16,"#94acbf")]:
        values = table.loc[table.Modell.eq(model)].set_index("Variable").VIF.reindex(labels)
        finite = np.isfinite(values)
        ax.barh(np.arange(len(labels))[finite]+offset,values[finite],height=.28,color=colour,label=model)
        for index in np.flatnonzero(np.isinf(values)):
            ax.text(1,index+offset,"∞",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=9)
    ax.set_yticks(range(len(labels)),[textwrap.fill(t,29) for t in labels],fontsize=9)
    ax.invert_yaxis()
    finite_values = table.loc[np.isfinite(table.VIF), "VIF"]
    upper = max(1., finite_values.max()) * 1.1 if len(finite_values) else 1.
    ax.set_xlim(0,upper)
    ticks = MaxNLocator(nbins=4).tick_values(0,upper)
    ax.set_xticks(ticks[(ticks>=0)&(ticks<=upper)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:german(x,0)))
    ax.set_xlabel("VIF")
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.03),ncol=2)
    apa7_axes(ax,remove_left=True)
    ax.tick_params(axis="y",length=0)
    save(fig,folder,"figure_5_vif")
    return fig
