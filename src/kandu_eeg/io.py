"""Data loading: raw .mat recordings + clinical table -> MNE objects.

The dataset (Liu et al., Scientific Data 2024, doi:10.1038/s41597-023-02787-8;
NEMAR nm000158; figshare 10.6084/m9.figshare.21679035) stores each patient as
`rawdata` = (40 trials, 33 channels, 4000 samples) at 500 Hz. The 40 trials were
recorded back-to-back (8 s each, no gap), so we concatenate them into one
continuous 320-s recording. Filtering a continuous signal avoids the edge
artefacts you get when you filter short epochs.
"""
from __future__ import annotations

import re

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from . import config as C


# --------------------------------------------------------------------------- #
# Clinical / demographic table
# --------------------------------------------------------------------------- #
def _lesion_side_from_text(loc: str) -> str:
    """Rough parse of the free-text lesion location -> 'left' | 'right' | 'bilateral' | 'unknown'."""
    s = loc.lower()
    left, right = "left" in s, "right" in s
    if left and right:
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    return "unknown"


_CORTICAL = re.compile(r"cortex|lobe|frontal|parietal|temporal|occipital|insula", re.I)


def load_participants() -> pd.DataFrame:
    """Clinical table with derived columns used throughout the analysis.

    Derived columns
    ---------------
    subject             int 1..50
    affected_hand       'left' / 'right' (= ParalysisSide)
    contralesional_hemi hemisphere *ipsilateral* to the affected hand: that is the
                        healthy hemisphere Kandu/IpsiHand records from. We derive it
                        from the paralysis side (what a clinician would use), because
                        many lesions are in the brainstem where "side" is ambiguous.
    lesion_side_text    side parsed from the free-text location
    side_concordant     True if lesion side is opposite to paralysis side (expected)
    cortical_lesion     True if the lesion text mentions cortex / a lobe
    """
    df = pd.read_csv(C.PARTICIPANTS_TSV, sep="\t")
    df["subject"] = df["Participant_ID"].str.replace("sub-", "").astype(int)
    df["affected_hand"] = df["ParalysisSide"].str.strip().str.lower()
    df["contralesional_hemi"] = df["affected_hand"]          # same side as the paretic hand
    df["lesion_side_text"] = df["StrokeLocation"].map(_lesion_side_from_text)
    opposite = {"left": "right", "right": "left"}
    df["side_concordant"] = df.apply(
        lambda r: r["lesion_side_text"] == opposite[r["affected_hand"]], axis=1
    )
    df["cortical_lesion"] = df["StrokeLocation"].str.contains(_CORTICAL)
    return df.set_index("subject")


# --------------------------------------------------------------------------- #
# EEG
# --------------------------------------------------------------------------- #
def raw_path(subject: int):
    return C.RAW_DIR / f"sub-{subject:02d}" / f"sub-{subject:02d}_task-motor-imagery_eeg.mat"


def load_mat(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rawdata [trials, 33, samples] in µV as stored, labels [trials])."""
    m = sio.loadmat(raw_path(subject))["eeg"][0, 0]
    return m["rawdata"].astype(np.float64), m["label"].ravel().astype(int)


def load_raw(subject: int, keep_ref: bool = False) -> mne.io.BaseRaw:
    """Load one patient as an *unfiltered* MNE Raw object made of 40 segments.

    - Units converted µV -> V (MNE convention: SI units).
    - Channel types: 29 EEG, 2 EOG (+ CPz reference if keep_ref, as a flat channel).
    - Standard 10-20 montage attached (needed for topomaps, Laplacian, interpolation).
    - Each 8-s trial block is a separate segment joined with an "EDGE boundary"
      annotation. WHY: the blocks are NOT perfectly contiguous in time (we measured
      2-9x larger filter transients at block joins than mid-block in most patients),
      so filtering across a join would smear a step artefact into the data. MNE's
      filters automatically process each EDGE-delimited segment separately.
    - Trigger -> annotations ('instruction/<hand>', 'mi/<hand>', 'break/<hand>').
      Trials whose MI marker is in the first 0.5 s of a block are flagged
      'BAD_no_baseline' and excluded (no pre-cue data to use as rest/reference).
    """
    X, labels = load_mat(subject)
    n_tr, n_ch, n_s = X.shape
    sf = C.SFREQ_RAW
    ch_names = [C.RENAME_1010.get(c, c) for c in C.MAT_CHANNELS][:-1]
    keep = [i for i, c in enumerate(ch_names) if keep_ref or c != C.REF_CHANNEL]
    names = [ch_names[i] for i in keep]
    types = ["eog" if c in C.EOG_CHANNELS else "eeg" for c in names]
    info = mne.create_info(names, sf, types)
    info["line_freq"] = C.LINE_FREQ

    data = X[:, keep, :].transpose(1, 0, 2).reshape(len(keep), n_tr * n_s) * 1e-6  # µV -> V
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")

    # Trigger channel: 1-sample pulses 1 = instruction, 2 = MI onset, 3 = break,
    # plus spurious non-integer glitch values. Keep the first EXACT 1/2/3 per block.
    on, dur, desc = [], [], []
    for k in range(1, n_tr):
        on.append(k * n_s / sf), dur.append(0.0), desc.append("EDGE boundary")
    for k in range(n_tr):
        block = X[k, -1]
        lab = C.LABEL_CODES[labels[k]]
        first = {c: (np.flatnonzero(block == c)[0] if np.any(block == c) else None) for c in (1, 2, 3)}
        mi = first[2]
        if mi is None:
            continue
        if mi < int(0.5 * sf):            # MI at block start -> no baseline in this segment
            on.append(k * n_s / sf), dur.append(n_s / sf), desc.append("BAD_no_baseline")
            continue
        brk = first[3] if (first[3] is not None and first[3] > mi) else mi + int(4 * sf)
        if first[1] is not None and first[1] < mi:
            on.append((k * n_s + first[1]) / sf), dur.append((mi - first[1]) / sf)
            desc.append(f"instruction/{lab}")
        on.append((k * n_s + mi) / sf), dur.append((brk - mi) / sf), desc.append(f"mi/{lab}")
        on.append((k * n_s + brk) / sf), dur.append(max(0.0, (n_s - brk) / sf)), desc.append(f"break/{lab}")
    raw.set_annotations(mne.Annotations(on, dur, desc))
    raw.info["subject_info"] = {"his_id": f"sub-{subject:02d}"}
    return raw


def segments(raw: mne.io.BaseRaw) -> list[tuple[int, int]]:
    """(start, stop) sample indices of EDGE-delimited segments."""
    a = raw.annotations
    edges = [int(round(o * raw.info["sfreq"])) for o, d in zip(a.onset, a.description) if d.startswith("EDGE")]
    b = sorted(set([0] + edges + [raw.n_times]))
    return [(x, y) for x, y in zip(b[:-1], b[1:]) if y - x > 1]


def mi_events(raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict]:
    """Events array at MI onsets (code 2) -> {'left_hand': 1, 'right_hand': 2}."""
    ev_id = {"mi/left_hand": 1, "mi/right_hand": 2}
    events, _ = mne.events_from_annotations(raw, event_id=ev_id, verbose="ERROR")
    return events, {"left_hand": 1, "right_hand": 2}
