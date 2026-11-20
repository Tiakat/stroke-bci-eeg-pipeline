"""Plot helpers: one consistent style for every figure (screen-share friendly)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np

# Fixed categorical order - colour follows the entity, never its rank.
CL_COLOR = "#2a78d6"     # contralesional (healthy) hemisphere  -> blue
IL_COLOR = "#eb6834"     # ipsilesional (lesioned) hemisphere   -> orange
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = "RdBu_r"     # ERD (power decrease) = blue, ERS = red, 0 = white


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10.5,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def info_for(ch_names: list[str], sfreq: float = 250.0) -> mne.Info:
    info = mne.create_info(ch_names, sfreq, "eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    return info


def topomap(values: dict[str, float], ax, vlim=None, cmap=DIVERGING, title=None, mask_chs=None,
            names=False):
    """Topomap from {channel: value}."""
    chs = [c for c in values if not np.isnan(values[c])]
    data = np.array([values[c] for c in chs])
    info = info_for(chs)
    if vlim is None:
        m = np.nanmax(np.abs(data))
        vlim = (-m, m)
    mask = np.array([c in (mask_chs or []) for c in chs])
    im, _ = mne.viz.plot_topomap(
        data, info, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=4,
        mask=mask if mask_chs else None, names=chs if names else None,
        mask_params=dict(marker="o", markerfacecolor="white", markeredgecolor="k", markersize=7),
        sensors=True,
    )
    if title:
        ax.set_title(title)
    return im


def savefig(fig, name: str):
    from .config import FIG_DIR

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
