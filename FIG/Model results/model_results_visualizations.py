"""Chapter 5 figures from the portable model exports; no model estimation."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from FIG.plotting.apa7 import apa7_axes, set_apa7_style


RESOLUTIONS = (100, 200, 500)
GRID_STYLE = {
    100: dict(color="#245477", marker="o", markersize=5.5, elinewidth=1.6),
    200: dict(color="#7496b0", marker="s", markersize=4.5, elinewidth=1.0),
    500: dict(color="#a1b7c9", marker="^", markersize=4.5, elinewidth=1.0),
}
SECTORS = {
    "1": "Gewerbe und Handwerk", "2": "Industrie", "3": "Handel",
    "4": "Bank und Versicherung", "5": "Transport und Verkehr",
    "6": "Tourismus und Freizeitwirtschaft", "7": "Information und Consulting",
}
FIGURE_NAMES = {
    "R1": "figure_R1_nachfrage_unternehmensumfeld",
    "R2": "figure_R2_nachfragegruppen",
    "R3": "figure_R3_oev_erschliessung",
    "R4": "figure_R4_autobahn_sparten",
    "R5": "figure_R5_austrittsrisiko",
}
REQUIRED_FILES = (
    "laufinfo.json", "koeffizienten.csv", "hypothesentests.csv", "diagnostik.csv",
    "stichprobe.csv", "plotdaten.parquet", "kovarianzen.parquet",
)
CSV_COLUMNS = {
    "label": "Merkmal", "resolution": "Rasterweite", "estimate": "Schätzwert",
    "lower": "Untergrenze", "upper": "Obergrenze", "run_id": "Lauf-ID",
    "status": "Status", "role": "Darstellung",
}


def set_model_results_style():
    """Use the shared figure style with Arial required for every model figure."""
    font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    set_apa7_style()
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"]})


def german_number(value, _position=None):
    return f"{value:.4g}".replace(".", ",")


def ratio_axis(ax, bounds):
    """Log axis with readable decimal ticks, including the null effect at one."""
    low, high = min(bounds[0], 1.0), max(bounds[1], 1.0)
    span = max(np.log(high) - np.log(low), 0.06)
    low, high = np.exp(np.log(low) - span * .12), np.exp(np.log(high) + span * .12)
    step = 10 ** np.floor(np.log10((high - low) / 5))
    for multiple in (1, 2, 2.5, 5, 10):
        spacing = step * multiple
        ticks = np.arange(np.ceil(low / spacing), np.floor(high / spacing) + 1) * spacing
        if len(ticks) <= 6:
            break
    ticks = np.unique(np.r_[ticks, 1.0])
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(german_number))
    ax.axvline(1, color=".4", linestyle="--", linewidth=.8, zorder=1)
    apa7_axes(ax, remove_left=True)
    ax.tick_params(axis="y", left=False, labelleft=False)


def load_bundles(project_dir, run_names):
    """Only explicitly selected final runs, with the same analysis specification."""
    bundles, notes = {}, []
    for resolution in RESOLUTIONS:
        folder = Path(project_dir) / "RESULTS" / f"{resolution}m" / run_names[resolution]
        missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
        if missing:
            notes.append(f"{resolution} m fehlt: Ergebnisdateien fehlen ({', '.join(missing)}).")
            continue
        try:
            info = json.loads((folder / "laufinfo.json").read_text(encoding="utf-8"))
            if info.get("status") != "completed" or info.get("resolution_m") != resolution:
                raise ValueError("Lauf nicht abgeschlossen oder falsche Rasterweite")
            if info.get("run_name") != run_names[resolution]:
                raise ValueError("Laufname stimmt nicht mit der expliziten Auswahl überein")
            bundle = dict(info=info, folder=folder)
            for name, filename in [("coefficients", "koeffizienten.csv"),
                                   ("tests", "hypothesentests.csv"),
                                   ("diagnostics", "diagnostik.csv"),
                                   ("sample", "stichprobe.csv")]:
                bundle[name] = pd.read_csv(folder / filename)
            bundle["plotdata"] = pd.read_parquet(folder / "plotdaten.parquet")
            bundle["covariances"] = pd.read_parquet(folder / "kovarianzen.parquet")
            if bundle["coefficients"].duplicated(["model", "term"]).any():
                raise ValueError("doppelte Modellkoeffizienten")
            if bundle["covariances"].duplicated(["model", "covariance", "term_i", "term_j"]).any():
                raise ValueError("doppelte Kovarianzeinträge")
            # Explicitly require the final H4 specification, rather than using an old run.
            terms = set(bundle["coefficients"].loc[
                bundle["coefficients"].model.eq("founding_nb2"), "term"])
            if not {"walk_pt_routes_10min", "log_walk_pt_departures_10min",
                    "pt_ohne_haltestelle"}.issubset(terms):
                raise ValueError("endgültige H4-Spezifikation fehlt")
            bundle["run_id"] = (
                f"{resolution}m/{info['run_name']}@{info.get('created_at', 'Zeit unbekannt')}"
            )
            bundles[resolution] = bundle
        except (ValueError, OSError, KeyError) as error:
            notes.append(f"{resolution} m ausgelassen: {error}.")
    if bundles:
        reference = bundles[min(bundles)]["info"]
        for resolution, bundle in list(bundles.items()):
            info = bundle["info"]
            keys = ("pipeline_version", "study_period", "time_aggregation")
            if any(info.get(key) != reference.get(key) for key in keys):
                notes.append(f"{resolution} m ausgelassen: Spezifikation passt nicht zum Referenzlauf.")
                del bundles[resolution]
    return bundles, notes


def model_problem(bundle, model):
    """The estimator exports check=fit/status=estimated only after a successful fit."""
    diagnostics = bundle["diagnostics"]
    rows = diagnostics.loc[diagnostics.model.eq(model) & diagnostics.check.eq("fit")]
    if len(rows) != 1 or rows.iloc[0].status != "estimated":
        return "kein erfolgreich abgeschlossener Fit im Diagnoseexport"
    if rows.iloc[0].get("specification") != "main":
        # The sensitivity models deliberately carry their own specification.
        if model in {"founding_nb2", "survival_cox"} or model.startswith(("h3_", "h5_")):
            return "keine Hauptspezifikation"
    detail = str(rows.iloc[0].get("detail", "")).lower()
    if any(word in detail for word in ("converg", "maximum number", "iteration limit")):
        return "Konvergenzwarnung im Diagnoseexport"
    if not bundle["covariances"].loc[
        bundle["covariances"].model.eq(model) & bundle["covariances"].covariance.eq("cell")
    ].shape[0]:
        return "Zellcluster-Kovarianz fehlt"
    return None


def single_row(frame, **filters):
    for column, value in filters.items():
        frame = frame.loc[frame[column].eq(value)]
    if len(frame) > 1:
        raise ValueError(f"Mehrdeutiger Export: {filters}")
    return None if frame.empty else frame.iloc[0]


def hypothesis(bundle, model, identifier, **filters):
    return single_row(bundle["tests"], model=model, hypothesis_id=identifier,
                      covariance="cell", specification="main", **filters)


def coefficient(bundle, model, term):
    return single_row(bundle["coefficients"], model=model, term=term, specification="main")


def prepare_chapter_data(bundles, notes):
    """Exponentiate exported estimates and CI endpoints, never reconstruct contrasts."""
    records = []

    def add(bundle, resolution, figure, key, label, expected, model, row,
            estimate_column="estimate", factor=1, block="", role="Punkt"):
        problem = model_problem(bundle, model)
        if problem is None and row is None:
            problem = "Merkmal oder Kontrast fehlt im Export"
        values = np.full(3, np.nan)
        if problem is None:
            values = np.array([row[estimate_column], row.ci_lower, row.ci_upper], dtype=float)
            if not np.isfinite(values).all() or not values[1] <= values[0] <= values[2]:
                problem = "Schätzwert oder Intervall nicht endlich oder ungültig"
            else:
                values = np.exp(values * factor)
                if not np.isfinite(values).all() or (values <= 0).any():
                    problem = "exponentiertes Intervall außerhalb des darstellbaren Bereichs"
        if problem:
            values[:] = np.nan
            notes.append(f"{figure}, {resolution} m, {label}: {problem}; nicht schätzbar.")
        records.append(dict(figure=figure, key=key, label=label, expected=expected,
                            model=model, resolution=resolution, run_id=bundle["run_id"],
                            estimate=values[0], lower=values[1], upper=values[2],
                            status="nicht schätzbar" if problem else "dargestellt",
                            block=block, role=role))

    for resolution, bundle in bundles.items():
        add(bundle, resolution, "R1", "H1", "Bevölkerung im Autoumfeld, bei gleichem Bestand",
            "H1 · erwartet > 1", "founding_nb2", hypothesis(bundle, "founding_nb2", "H1"))
        add(bundle, resolution, "R1", "H2a", "Unternehmensbestand im Autoumfeld, bei gleicher Bevölkerung",
            "H2a · erwartet > 1", "founding_nb2", hypothesis(bundle, "founding_nb2", "H2a"))
        add(bundle, resolution, "R1", "own_pop", "Bevölkerung der eigenen Rasterzelle (Vorjahr)",
            "Kontrolle", "founding_nb2", coefficient(bundle, "founding_nb2", "log_own_pop"),
            "coefficient", block="Kontrolle")
        groups = bundle["plotdata"].loc[bundle["plotdata"].plot_type.eq("h3_population")]
        for group, description in [("A", "hohe"), ("C", "geringe")]:
            row = single_row(groups, group=group)
            count = "unbekannt" if row is None else f"{int(row.births):,}".replace(",", ".")
            label = f"Gruppe {group}: {description} lokale Nachfrageabhängigkeit (n = {count} Gründungen)"
            add(bundle, resolution, "R2", group, label, "erwartet > 1", "h3_" + group,
                row, "population_effect")
        add(bundle, resolution, "R2", "H3", "Differenz A gegenüber C", "H3 · erwartet > 1",
            "h3_A", hypothesis(bundle, "h3_A_vs_C", "H3"), block="Gruppenvergleich")
        # The difference requires both matched group models to have converged.
        if model_problem(bundle, "h3_C"):
            records[-1].update(estimate=np.nan, lower=np.nan, upper=np.nan, status="nicht schätzbar")
            notes.append(f"R2, {resolution} m: H3 fehlt, weil Gruppe C nicht erfolgreich geschätzt wurde.")
        for key, label, expected, factor in [
            ("H4_routes", "Linienzahl in 10 Gehminuten, je 10 Linien", "erwartet > 1", 10),
            ("H4_frequency", "Abfahrtsfrequenz in 10 Gehminuten (logarithmiert)", "erwartet > 1", 1),
            ("H4_no_stop", "Keine Haltestelle in 10 Gehminuten", "erwartet < 1", 1),
        ]:
            add(bundle, resolution, "R3", key, label, expected, "founding_nb2",
                hypothesis(bundle, "founding_nb2", key), factor=factor)
        for key, identifier, label, expected, block in [
            ("same_ring", "H2b", "Branchengleicher Unternehmensbestand", "H2b · erwartet > 1", "Autoumfeld (15 Minuten)"),
            ("other_ring", "H2c", "Branchenfremder Unternehmensbestand", "H2c · erwartet < 1", "Autoumfeld (15 Minuten)"),
            ("same_own", "log_own_same", "Branchengleicher Unternehmensbestand", "Kontrolle", "Eigene Rasterzelle"),
            ("other_own", "log_own_other", "Branchenfremder Unternehmensbestand", "Kontrolle", "Eigene Rasterzelle"),
        ]:
            is_control = expected == "Kontrolle"
            row = coefficient(bundle, "survival_cox", identifier) if is_control else hypothesis(bundle, "survival_cox", identifier)
            add(bundle, resolution, "R5", key, label, expected, "survival_cox", row,
                "coefficient" if is_control else "estimate", block=block)
        if resolution != 100:
            continue
        diagnostics = bundle["diagnostics"]
        for sector, name in SECTORS.items():
            coverage = diagnostics.loc[diagnostics.check.eq("sector_coverage") &
                                       diagnostics.sparte.astype(str).isin([sector, sector + ".0"])]
            births = coverage.births_retained.sum(min_count=1)
            n_label = "unbekannt" if pd.isna(births) else f"{int(births):,}".replace(",", ".")
            add(bundle, resolution, "R4", sector, f"{name} (n = {n_label})", "", "h5_" + sector,
                coefficient(bundle, "h5_" + sector, "log_tt_motorway_exit"), "coefficient")
        add(bundle, resolution, "R4", "overall", "Gesamtmodell", "", "founding_nb2",
            coefficient(bundle, "founding_nb2", "log_tt_motorway_exit"), "coefficient", role="Referenzband")
    columns = ["figure", "key", "label", "expected", "model", "resolution", "run_id",
               "estimate", "lower", "upper", "status", "block", "role"]
    return pd.DataFrame(records, columns=columns)


def save_figure(fig, folder, name):
    """Keep the final width at 16 cm; explicit margins avoid variable-size exports."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        for extension in ("png", "svg"):
            fig.savefig(folder / f"{name}.{extension}", dpi=300, facecolor="white")


def export_values(table, folder, name):
    table.rename(columns=CSV_COLUMNS)[list(CSV_COLUMNS.values())].to_csv(
        Path(folder) / f"{name}.csv", index=False, encoding="utf-8-sig")


def legend_handles(resolutions):
    return [Line2D([], [], linestyle="none", color=GRID_STYLE[size]["color"],
                   marker=GRID_STYLE[size]["marker"], markersize=GRID_STYLE[size]["markersize"],
                   markerfacecolor="white" if size == 500 else GRID_STYLE[size]["color"],
                   label=f"{size} m" + (" (Hauptmodell)" if size == 100 else ""))
            for size in resolutions]


def draw_forest(table, figure_id, folder, name=None):
    """Label column and plotting column keep long German text readable at 9 pt."""
    if table.empty:
        return None
    set_model_results_style()
    points = table.loc[table.role.eq("Punkt")]
    reference = table.loc[table.role.eq("Referenzband") & table.status.eq("dargestellt")]
    rows = points.drop_duplicates("key")
    available = sorted(points.loc[points.status.eq("dargestellt"), "resolution"].unique())
    wrapped_labels = {}
    for row in rows.itertuples():
        wrapped = textwrap.fill(row.label, width=37)
        if figure_id == "R2":
            counts = points.loc[points.key.eq(row.key), ["resolution", "label"]].drop_duplicates()
            if counts.label.nunique() > 1:
                base = row.label.split(" (n =")[0]
                count_text = "; ".join(f"{r.resolution} m: {r.label.split('(n = ')[1].rstrip(')')}" for r in counts.itertuples())
                wrapped = textwrap.fill(base, 37) + "\n" + textwrap.fill(count_text, 37)
        wrapped_labels[row.key] = wrapped
    row_positions, y, last_block, headings, separators = {}, 0., "", [], []
    for row in rows.itertuples():
        if row.block and row.block != last_block:
            if y:
                separators.append(y - .30)
            headings.append((y, row.block))
            y += .55
        row_positions[row.key] = y
        line_count = wrapped_labels[row.key].count("\n") + 1
        y += max(1., (line_count * 10 + 7) / 35)
        last_block = row.block
    bottom_cm = 1.65
    top_cm = 1.25
    plot_cm = y * 1.25
    height_cm = plot_cm + bottom_cm + top_cm
    height = height_cm / 2.54
    fig = plt.figure(figsize=(16 / 2.54, height))
    label_ax = fig.add_axes([.025, bottom_cm / height_cm, .53, plot_cm / height_cm])
    ax = fig.add_axes([.59, bottom_cm / height_cm, .38, plot_cm / height_cm])
    for panel in (label_ax, ax):
        panel.set_ylim(y - .45, -.5)
    label_ax.set_xlim(0, 1)
    label_ax.set_axis_off()
    for position, heading in headings:
        label_ax.text(0, position - .13, heading, fontsize=9, weight="bold", va="center")
    for row in rows.itertuples():
        position = row_positions[row.key]
        wrapped = wrapped_labels[row.key]
        label_ax.annotate(wrapped, (0, position), xytext=(0, 0),
                          textcoords="offset points", va="center", fontsize=9,
                          weight="bold" if row.key == "H3" else "normal", linespacing=1.1)
        these = points.loc[points.key.eq(row.key)]
        if these.status.eq("nicht schätzbar").all():
            ax.text(.5, position, "nicht schätzbar", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9, color=".4")
        offsets = {size: offset for size, offset in zip(available, np.linspace(-.19, .19, len(available)) if len(available) > 1 else [0])}
        for item in these.loc[these.status.eq("dargestellt")].itertuples():
            style = GRID_STYLE[item.resolution]
            ax.errorbar(item.estimate, position + offsets[item.resolution],
                        xerr=[[item.estimate - item.lower], [item.upper - item.estimate]],
                        fmt=style["marker"], color=style["color"], markersize=style["markersize"],
                        elinewidth=style["elinewidth"], capsize=2.5,
                        markerfacecolor="white" if item.resolution == 500 else style["color"], zorder=3)
    for separator in separators:
        for panel in (label_ax, ax):
            panel.axhline(separator, color=".8", linewidth=.6)
    valid = table.loc[table.status.eq("dargestellt")]
    bounds = (.9, 1.1) if valid.empty else (valid.lower.min(), valid.upper.max())
    ratio_axis(ax, bounds)
    if not reference.empty:
        row = reference.iloc[0]
        ax.axvspan(row.lower, row.upper, color=".9", zorder=0)
        ax.axvline(row.estimate, color=".55", linewidth=1, zorder=1)
        ax.annotate("Gesamtmodell", (row.estimate, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=".4")
    hazard = figure_id == "R5"
    axis_label = "Hazard Ratio" if hazard else "Inzidenzratenverhältnis"
    if figure_id == "comparison":
        axis_label = "IRR / Hazard Ratio"
    ax.set_xlabel(axis_label + "\n95-%-Konfidenzintervall", fontsize=9)
    if available:
        fig.legend(handles=legend_handles(available), loc="upper center", ncol=3,
                   bbox_to_anchor=(.5, .995), fontsize=9, columnspacing=1.3, handletextpad=.4)
    name = name or FIGURE_NAMES[figure_id]
    save_figure(fig, folder, name)
    export_values(table, folder, name)
    return fig


def compact_comparison(data, folder):
    """A compact overview with explicit IRR and HR blocks and the available grids."""
    selections = [("R1", "H1", "Bevölkerung im Autoumfeld"),
                  ("R1", "H2a", "Unternehmensbestand im Autoumfeld"),
                  ("R2", "H3", "Nachfragegruppen A gegenüber C"),
                  ("R3", "H4_routes", "ÖV-Linienzahl, je 10 Linien"),
                  ("R3", "H4_frequency", "ÖV-Abfahrtsfrequenz (log.)"),
                  ("R3", "H4_no_stop", "Keine Haltestelle"),
                  ("R5", "same_ring", "Branchengleicher Bestand im Umfeld"),
                  ("R5", "other_ring", "Branchenfremder Bestand im Umfeld")]
    frames = []
    for figure, key, label in selections:
        frame = data.loc[data.figure.eq(figure) & data.key.eq(key)].copy()
        frame["label"] = label
        frame["block"] = "Austrittsrisiko · Hazard Ratio" if figure == "R5" else "Gründungen · Inzidenzratenverhältnis"
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    # Both quantities have a null effect at 1; each block is explicitly labelled.
    return draw_forest(table, "comparison", folder, "figure_kompakter_rastervergleich")


def write_protocol(bundles, notes, data, folder):
    lines = ["Begleitprotokoll: Kapitel 5, R1–R5", ""]
    for size in RESOLUTIONS:
        bundle = bundles.get(size)
        lines.append(f"{size} m: " + (bundle["run_id"] if bundle else "fehlt / nicht freigegeben"))
    lines += ["", "Quellen und Umrechnungen:",
              "Nur explizit ausgewählte, abgeschlossene Läufe derselben Pipeline/Spezifikation; kein Rückgriff auf ältere Läufe.",
              "Ein Modell gilt als erfolgreich geschätzt, wenn diagnostik.csv check=fit/status=estimated ausweist und keine Konvergenzwarnung enthält.",
              "Alle dargestellten Intervalle: 95 %, Zellcluster. Blockcluster werden nicht dargestellt.",
              "IRR/HR: exp(Schätzwert), exp(Untergrenze), exp(Obergrenze).",
              "R1 H1, H2a; R2 H3; R3 H4; R5 H2b/H2c: direkt aus hypothesentests.csv (covariance=cell).",
              "R2 Gruppen A/C: exportierte Bevölkerungskontraste und Zellclusterintervalle aus plotdaten.parquet (h3_population).",
              "R3 Linienzahl: Schätzwert und beide Log-Intervallgrenzen × 10 vor exp; Effekt je 10 Linien.",
              "R4: Koeffizienten aus Poisson-QMLE-Spartenmodellen; graues Referenzband aus dem NB2-Hauptmodell (100 m).",
              "R4 Gründungszahlen: diagnostik.csv, sector_coverage/births_retained; R2 aus den jeweiligen Gruppenmodellen.",
              "R2 Differenz und Rastervergleich: exp(H3) ist das Verhältnis zweier IRR, keine eigene Gründungsrate.",
              "CSV: exponentierte dargestellte Werte; R4 enthält das Referenzband und die nicht schätzbaren Zeilen mit leeren Werten."]
    captions = {
        "R1": "Nachfrage und Unternehmensumfeld im Gründungsmodell. Inzidenzratenverhältnisse mit 95-%-Konfidenzintervallen, Standardfehler nach Zellen geclustert.",
        "R2": "Bevölkerungseffekte nach lokaler Nachfrageabhängigkeit (Poisson-QMLE je Gruppe). Die unterste Zeile zeigt das Verhältnis der IRR von Gruppe A und C; 1 bedeutet keinen Unterschied.",
        "R3": "Fußläufige ÖV-Erschließung im Gründungsmodell. Linienzahl je 10 Linien.",
        "R4": "Autobahnfahrzeit nach Sparte (Poisson-QMLE je Sparte, 100 m). Graue Linie und Band: NB2-Gesamtmodell und dessen 95-%-Konfidenzintervall.",
        "R5": "Unternehmensumfeld und Austrittsrisiko im Cox-Modell. Hazard Ratios mit 95-%-Konfidenzintervallen, Standardfehler nach Zellen geclustert.",
    }
    if 100 in bundles:
        bundle = bundles[100]
        for model, identifier, figure in [("founding_nb2", "H4_joint", "R3"),
                                          ("h5_sectors", "H5_heterogeneity", "R4")]:
            row = hypothesis(bundle, model, identifier)
            required = ["founding_nb2"] if figure == "R3" else [f"h5_{s}" for s in SECTORS if not model_problem(bundle, f"h5_{s}")]
            if row is not None and pd.notna(row.p_value) and required and all(not model_problem(bundle, m) for m in required):
                detail = f"p = {row.p_value:.6g}".replace(".", ",")
                if figure == "R4":
                    detail += f"; {int(row.sectors_tested)} von 7 Sparten; Abdeckung: {row.coverage}"
                lines.append(f"{figure}, Wald-Test (100 m, Zellcluster): {detail}.")
                captions[figure] += f" Gemeinsamer Wald-Test: {detail}."
            else:
                lines.append(f"{figure}, Wald-Test (100 m): nicht verfügbar.")
    if 100 not in bundles:
        notes.append("R4 fehlt: kein freigegebener 100-m-Lauf; andere Rasterweiten werden dafür nicht verwendet.")
    lines += ["", "Fehlende Varianten und Prüfbedarf:", *dict.fromkeys(notes)]
    for size, bundle in bundles.items():
        flags = bundle["diagnostics"].loc[bundle["diagnostics"].status.isin(["REVIEW", "FAILED", "unidentified"])]
        for row in flags.itertuples():
            sector = f", {row.sparte_name}" if pd.notna(getattr(row, "sparte_name", np.nan)) else ""
            lines.append(f"{size} m, {row.model}{sector}, {row.check} [{row.status}]: {row.detail}")
    folder = Path(folder)
    (folder / "begleitprotokoll.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    available = ", ".join(f"{s} m" for s in bundles) or "keine"
    text = ["# Vorschläge für Word-Bildunterschriften", "",
            f"Verfügbare Rasterweiten: {available}. Fehlende Varianten siehe Begleitprotokoll.", ""]
    text += [f"**{key}:** {caption} Quelle: eigene Berechnung.\n" for key, caption in captions.items()]
    (folder / "bildunterschriften.md").write_text("\n".join(text), encoding="utf-8")


def create_chapter_figures(data, folder, resolutions=None):
    selected = data if resolutions is None else data.loc[data.resolution.isin(resolutions)]
    figures = []
    for identifier in FIGURE_NAMES:
        fig = draw_forest(selected.loc[selected.figure.eq(identifier)], identifier, folder)
        if fig is not None:
            figures.append(fig)
    return figures
