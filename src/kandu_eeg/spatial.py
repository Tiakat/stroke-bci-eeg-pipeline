"""Spatial operations: hemisphere mapping and re-referencing.

Re-referencing is the "5 electrodes reduce noise, 1 electrode is used" part of a
Kandu-like design. EEG is always a *difference* of potentials. Choosing what you
subtract from the control electrode decides what noise is cancelled:

    raw (vs CPz)  : C - CPz            -> CPz is ~2 cm from C3/C4: partly cancels motor activity
    CAR           : C - mean(all chans) -> removes everything common to all sensors
                                          (mains, movement, global drifts)
    Laplacian     : C - mean(4 neighbours) -> spatial high-pass, keeps local (focal) activity
    bipolar       : C - one neighbour   -> 2 electrodes only
"""
from __future__ import annotations

import numpy as np

from . import config as C

_LEFT = {l: r for l, r in C.HOMOLOGOUS_PAIRS}
_RIGHT = {r: l for l, r in C.HOMOLOGOUS_PAIRS}

# generic site -> (left name, right name)
_GENERIC = {
    "Fp": ("Fp1", "Fp2"), "F7/8": ("F7", "F8"), "F": ("F3", "F4"), "FT": ("FT7", "FT8"),
    "FC": ("FC3", "FC4"), "T": ("T7", "T8"), "C": ("C3", "C4"), "TP": ("TP7", "TP8"),
    "CP": ("CP3", "CP4"), "P7/8": ("P7", "P8"), "P": ("P3", "P4"), "O": ("O1", "O2"),
}


def resolve(site: str, contralesional_hemi: str) -> str:
    """'CL:C' -> 'C3' if the contralesional hemisphere is left, 'C4' if right.
    'IL:C' -> the opposite hemisphere. Plain names ('Cz', 'F3') are returned as-is."""
    if ":" not in site:
        return site
    side, gen = site.split(":")
    left, right = _GENERIC[gen]
    hemi = contralesional_hemi if side == "CL" else ("left" if contralesional_hemi == "right" else "right")
    return left if hemi == "left" else right


def montage_channels(name_or_list, contralesional_hemi: str) -> list[str]:
    sites = C.MONTAGES[name_or_list] if isinstance(name_or_list, str) else name_or_list
    return [resolve(s, contralesional_hemi) for s in sites]


def lesion_relative_names(ch_names: list[str], contralesional_hemi: str) -> dict[str, str]:
    """Rename channels to 'CL:<site>' / 'IL:<site>' / midline. Lets us average
    patients with left and right lesions together (mirroring)."""
    out = {}
    for gen, (l, r) in _GENERIC.items():
        cl, il = (l, r) if contralesional_hemi == "left" else (r, l)
        if cl in ch_names:
            out[cl] = f"CL:{gen}"
        if il in ch_names:
            out[il] = f"IL:{gen}"
    for m in C.MIDLINE:
        if m in ch_names:
            out[m] = m
    return out


def mirror_topography(values: dict[str, float], contralesional_hemi: str) -> dict[str, float]:
    """Flip a {channel: value} map so the contralesional hemisphere is always LEFT.
    Used to build group-average topomaps across left- and right-lesion patients."""
    if contralesional_hemi == "left":
        return dict(values)
    flipped = {}
    for ch, v in values.items():
        flipped[_RIGHT.get(ch, _LEFT.get(ch, ch))] = v
    return flipped


# --------------------------------------------------------------------------- #
# Re-referencing on numpy arrays (..., n_channels, n_times)
# --------------------------------------------------------------------------- #
def car(X: np.ndarray, idx_pool: list[int] | None = None) -> np.ndarray:
    """Common average reference computed over `idx_pool` channels (default: all)."""
    pool = X if idx_pool is None else X[..., idx_pool, :]
    return X - pool.mean(axis=-2, keepdims=True)


def laplacian(X: np.ndarray, ch_names: list[str], center: str, neighbours: list[str]) -> np.ndarray:
    """Small (Hjorth) Laplacian: centre minus mean of its neighbours -> (..., n_times)."""
    i = ch_names.index(center)
    nb = [ch_names.index(n) for n in neighbours]
    return X[..., i, :] - X[..., nb, :].mean(axis=-2)


def bipolar(X: np.ndarray, ch_names: list[str], a: str, b: str) -> np.ndarray:
    return X[..., ch_names.index(a), :] - X[..., ch_names.index(b), :]


def laplacian_neighbours(center: str) -> list[str]:
    """Nearest-neighbour ring for C3 / C4 available in this 29-channel cap."""
    table = {
        "C3": ["FC3", "CP3", "T7", "Cz"],
        "C4": ["FC4", "CP4", "T8", "Cz"],
    }
    return table[center]
