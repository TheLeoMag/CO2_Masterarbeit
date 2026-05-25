from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
OGD_DIR = PROJECT_DIR / "OGD"
FIG_DIR = PROJECT_DIR / "FIG"

DEFAULT_POPULATION = OGD_DIR / "pd_popreg_100m_7767c33f-302c-11e3-beb4-0000c1ab0db6.geoparquet"
DEFAULT_BOUNDARIES = OGD_DIR / "Gemeindegrenzen.zip"
DEFAULT_OUTPUT = FIG_DIR / "population_density_styria.png"

BOUNDARY_LAYER = "Gemeindegrenzen.shp"
BOUNDARY_COLUMNS = ["BEZNR6", "GEMNR", "GEMNAM"]
MAP_CRS = "EPSG:3035"


def read_zipped_shapefile(zip_path: Path, layer: str) -> gpd.GeoDataFrame:
    archive_path = zip_path.resolve().as_posix()
    zip_uri = f"zip:///{archive_path}" if not archive_path.startswith("/") else f"zip://{archive_path}"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*FLAECHE_HA parsed incompletely.*")
        return gpd.read_file(f"{zip_uri}!{layer}", columns=BOUNDARY_COLUMNS)


def load_styria_population(
    population_path: Path,
    boundaries_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoSeries]:
    municipalities = read_zipped_shapefile(boundaries_path, BOUNDARY_LAYER).to_crs(MAP_CRS)
    if hasattr(municipalities.geometry, "union_all"):
        styria_boundary = municipalities.geometry.union_all()
    else:
        styria_boundary = municipalities.geometry.unary_union

    population = gpd.read_parquet(population_path)
    if population.crs is None:
        population = population.set_crs(MAP_CRS)
    else:
        population = population.to_crs(MAP_CRS)

    minx, miny, maxx, maxy = municipalities.total_bounds
    population = population.cx[minx:maxx, miny:maxy]
    population = gpd.clip(population, styria_boundary)
    population = population[population["population"] > 0].copy()
    if population.empty:
        raise ValueError(
            "No population cells intersect the Styria boundary. Check that both inputs use the expected CRS."
        )

    return population, municipalities, gpd.GeoSeries([styria_boundary], crs=MAP_CRS)


def add_scale_bar(ax: plt.Axes, length_km: int = 25) -> None:
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    width = maxx - minx
    height = maxy - miny

    x0 = minx + width * 0.07
    y0 = miny + height * 0.06
    length_m = length_km * 1_000

    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=1.4)
    ax.plot([x0, x0], [y0 - height * 0.006, y0 + height * 0.006], color="black", linewidth=1.0)
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - height * 0.006, y0 + height * 0.006],
        color="black",
        linewidth=1.0,
    )
    ax.text(
        x0 + length_m / 2,
        y0 + height * 0.012,
        f"{length_km} km",
        ha="center",
        va="bottom",
        fontsize=8,
    )


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.16),
        xytext=(0.94, 0.07),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.1, "color": "black"},
    )


def add_patch_legend(ax: plt.Axes, colors, labels, title: str) -> None:
    handles = [Patch(facecolor=color, edgecolor="#4d4d4d", linewidth=0.3) for color in colors]
    legend = ax.legend(
        handles,
        labels,
        title=title,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.35,
        handlelength=1.0,
        handleheight=1.0,
        fontsize=8,
        title_fontsize=8,
    )
    legend.get_frame().set_linewidth(0.4)
    legend.get_frame().set_edgecolor("#808080")


def plot_population_density(
    population: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    styria_boundary: gpd.GeoSeries,
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 7.2), constrained_layout=True)
    ax.set_facecolor("white")

    municipalities.plot(ax=ax, facecolor="#f7f7f7", edgecolor="none", linewidth=0)

    population_bin_edges = [0, 5, 10, 25, 50, 100, 250, 500, 1_000]
    max_population = int(population["population"].max())
    if max_population >= population_bin_edges[-1]:
        population_bin_edges.append(max_population + 1)
    else:
        population_bin_edges = [edge for edge in population_bin_edges if edge <= max_population]
        population_bin_edges = [0, *population_bin_edges[1:], max_population + 1]

    n_population_bins = len(population_bin_edges) - 1
    population_color_positions = (
        [0.0]
        if n_population_bins == 1
        else [i / (n_population_bins - 1) for i in range(n_population_bins)]
    )
    population_cmap = ListedColormap(plt.get_cmap("RdYlGn_r")(population_color_positions))
    population_norm = BoundaryNorm(population_bin_edges, population_cmap.N)
    population_bin_labels = []
    for lower, upper in zip(population_bin_edges[:-1], population_bin_edges[1:]):
        if upper == max_population + 1 and lower >= 1_000:
            population_bin_labels.append(f"{lower:,}+")
        else:
            population_bin_labels.append(f"{max(1, lower):,}-{upper - 1:,}")
    population_legend_colors = population_cmap.colors

    population.plot(
        ax=ax,
        column="population",
        cmap=population_cmap,
        norm=population_norm,
        linewidth=0,
        rasterized=True,
    )

    municipalities.boundary.plot(ax=ax, color="#ffffff", linewidth=0.18, alpha=0.75)
    styria_boundary.boundary.plot(ax=ax, color="#1f1f1f", linewidth=0.7)

    ax.set_title("Population Distribution in Styria", fontsize=11, pad=8)
    ax.set_axis_off()
    ax.set_aspect("equal")
    add_patch_legend(ax, population_legend_colors, population_bin_labels, "Population per 100 m^2")
    add_north_arrow(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-ready heat map of population in Styria."
    )
    parser.add_argument(
        "--population",
        type=Path,
        default=DEFAULT_POPULATION,
        help="GeoParquet with extracted population grid cells.",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=DEFAULT_BOUNDARIES,
        help="Zip file containing the Styria municipality shapefile.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path. A PDF with the same basename is also written.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    population, municipalities, styria_boundary = load_styria_population(
        args.population.resolve(),
        args.boundaries.resolve(),
    )
    plot_population_density(
        population,
        municipalities,
        styria_boundary,
        args.output.resolve(),
        args.dpi,
    )
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {args.output.resolve().with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
