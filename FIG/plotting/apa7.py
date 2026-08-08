import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

APA7_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

DEFAULT_WIDTH = 6.5
HALF_WIDTH = 3.25
DEFAULT_HEIGHT = 4.0
MAP_HEIGHT = 5.0

# Reuse these defaults rather than repeating visual styling in notebooks.
BAR_KWARGS = {"color": "0.75", "edgecolor": "black", "linewidth": 0.8}
HIST_KWARGS = {"color": "0.75", "edgecolor": "black", "linewidth": 0.8}
LINE_KWARGS = {"color": "black", "linewidth": 1.5}
REFERENCE_LINE_KWARGS = {"color": "black", "linewidth": 0.8, "linestyle": "--"}
ERRORBAR_KWARGS = {"ecolor": "black", "elinewidth": 0.8, "capsize": 3}

def set_apa7_style():
    mpl.rcParams.update(APA7_STYLE)

def apa7_axes(ax, remove_top=True, remove_right=True, remove_left=False, remove_bottom=False):
    ax.spines["top"].set_visible(not remove_top)
    ax.spines["right"].set_visible(not remove_right)
    ax.spines["left"].set_visible(not remove_left)
    ax.spines["bottom"].set_visible(not remove_bottom)
    ax.tick_params(axis="both", which="both", direction="out")
    return ax

def new_figure(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    fig, ax = plt.subplots(figsize=(width, height))
    apa7_axes(ax)
    return fig, ax

def new_map_figure(width=DEFAULT_WIDTH, height=MAP_HEIGHT):
    """Create a map figure without statistical-chart axes."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()
    return fig, ax

def save_apa7(fig, filename, dpi=600):
    fig.savefig(Path(filename), dpi=dpi, bbox_inches="tight", facecolor="white")
