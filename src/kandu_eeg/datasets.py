"""Unified loaders: four open datasets -> one `Recording` format.

Each dataset was chosen because it isolates one aspect of the Kandu / IpsiHand use case:

| key        | population                  | task                                   | why it matters for Kandu                      |
|------------|-----------------------------|----------------------------------------|-----------------------------------------------|
| liu2024    | 50 acute stroke patients    | imagined hand grip (video-guided)      | stroke, clinical headset, lesion side known   |
| bnci2019   | 9 cervical SCI patients     | ATTEMPTED hand open / grasp            | attempted movement of a paralysed hand        |
| physionet  | healthy adults (n=49 here)  | EXECUTED and IMAGINED fist open/close  | executed vs imagined; normative laterality    |
| bnci2014   | 9 healthy adults            | imagined left / right hand (BCI IV-2a) | the standard benchmark (sanity check)         |

`Recording` holds an UNFILTERED continuous MNE Raw (volts, montage, EOG typed) plus
events time-locked to movement onset, the hand of each event, and the windows that
define MOVE and REST for that paradigm. Filtering is done by the common pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from . import config as C

EXT = C.PROJECT_ROOT / "data_ext"


@dataclass
class Recording:
    dataset: str
    subject: str
    raw: mne.io.BaseRaw
    events: np.ndarray                 # (n, 3) MNE events at movement / imagery onset
    event_id: dict                     # label -> code
    hand: dict                         # label -> 'left' | 'right'
    task: str                          # 'imagery' | 'attempt' | 'execution'
    move_win: tuple = (0.5, 2.0)       # s after onset
    rest_mode: str = "precue"          # 'precue' (window before onset) or 'events' (separate rest events)
    rest_win: tuple = (-1.5, 0.0)
    rest_events: np.ndarray | None = None
    population: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def sfreq(self):
        return self.raw.info["sfreq"]


# --------------------------------------------------------------------------- #
def _finish(raw: mne.io.BaseRaw, eog: list[str] | None = None, rename: dict | None = None):
    if rename:
        raw.rename_channels(rename)
    if eog:
        raw.set_channel_types({c: "eog" for c in eog if c in raw.ch_names})
    raw.set_montage("standard_1005", on_missing="ignore", match_case=False, verbose="ERROR")
    return raw


# --------------------------------------------------------------------------- liu2024
def load_liu2024(subject: int) -> Recording:
    """Acute stroke, imagined grip. Only AFFECTED-hand events are labelled as 'move'
    (and unaffected ones too, with their hand), contralesional side from the clinical table."""
    from . import io

    part = io.load_participants()
    raw = io.load_raw(subject)
    events, ev_id = io.mi_events(raw)
    aff = part.loc[subject, "affected_hand"]
    return Recording(
        "liu2024", f"{subject:02d}", raw, events, ev_id,
        hand={"left_hand": "left", "right_hand": "right"}, task="imagery",
        move_win=C.MI_WIN, rest_mode="precue", rest_win=C.REST_WIN,
        population="acute stroke",
        meta=dict(affected_hand=aff, nihss=part.loc[subject, "NIHSS"]),
    )


# --------------------------------------------------------------------------- bnci2019
BNCI2019_HAND_CODES = {"hand_open": 779, "palmar_grasp": 925, "lateral_grasp": 926}
BNCI2019_TESTED_HAND = {f"P{p:02d}": ("left" if p == 7 else "right") for p in range(1, 10)}


def load_bnci2019(subject: int, include_rest_runs: bool = True) -> Recording:
    """Ofner et al. 2019 (BNCI 2019-001): cervical SCI, ATTEMPTED movements of one hand.

    Runs 3-7, 10-13 = attempted movement (5 classes, we keep the 3 HAND classes);
    runs 2, 9, 15 = 70-s rest; runs 1, 8, 14 = eye-movement calibration.
    Fixation cross from -2 s, class cue at 0 s (attempt immediately). The pre-cue rest
    window shows the same fixation cross as the movement window -> visually matched.
    """
    pid = f"P{subject:02d}"
    folder = EXT / "bnci2019" / pid
    runs = [3, 4, 5, 6, 7, 10, 11, 12, 13] + ([2, 9, 15] if include_rest_runs else [])
    raws, rest_spans = [], []
    for r in runs:
        rr = mne.io.read_raw_gdf(folder / f"{pid} Run {r}.gdf", preload=True, verbose="ERROR")
        # Each GDF ends with a block of NaN samples (amplifier stop): crop it off.
        bad = np.flatnonzero(np.isnan(rr._data).any(0))
        if bad.size:
            rr.crop(0, (bad[0] - 1) / rr.info["sfreq"])
        # Values are stored in µV but the header unit says V -> rescale to volts.
        if np.median(np.std(rr._data, axis=1)) > 1e-3:
            rr._data *= 1e-6
        # 256 -> 128 Hz (integer factor, anti-aliased): halves memory (64 ch x ~50 min);
        # Nyquist 64 Hz is still far above our 40 Hz low-pass and the 3 Hz MRCP band.
        rr.resample(128.0, method="polyphase", verbose="ERROR")
        if r in (2, 9, 15):
            # mark the whole run as a rest segment
            rr.set_annotations(mne.Annotations([2.0], [rr.times[-1] - 4.0], ["rest_run"]))
        raws.append(rr)
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    raw = _finish(raw, eog=["eog-l", "eog-m", "eog-r"])

    ann = raw.annotations
    sf = raw.info["sfreq"]
    hand = BNCI2019_TESTED_HAND[pid]
    rows = []
    for o, d in zip(ann.onset, ann.description):
        if d.isdigit() and int(d) in BNCI2019_HAND_CODES.values():
            rows.append([int(round(o * sf)), 0, 1])
    events = np.array(rows, int)
    # rest events: 1.5-s windows tiled every 2 s inside the rest runs (step > window -> no overlap)
    rest = []
    for o, d, du in zip(ann.onset, ann.description, ann.duration):
        if d == "rest_run":
            for t in np.arange(o + 2.0, o + du - 2.0, 2.0):
                rest.append([int(round(t * sf)), 0, 99])
    return Recording(
        "bnci2019", pid, raw, events, {f"{hand}_hand": 1}, {f"{hand}_hand": hand}, task="attempt",
        move_win=(0.5, 2.0), rest_mode="precue", rest_win=(-1.5, 0.0),
        rest_events=np.array(rest, int) if rest else None, population="cervical SCI",
        meta=dict(tested_hand=hand),
    )


# --------------------------------------------------------------------------- physionet
def load_physionet(subject: int, task: str = "execution") -> Recording:
    """Schalk et al. 2004 (PhysioNet EEGMMIDB). Fist open/close of the left or right hand.

    task='execution' -> runs 3, 7, 11 ; task='imagery' -> runs 4, 8, 12.
    T1 = left fist, T2 = right fist (4.1 s target on screen), T0 = rest (blank screen).
    REST = T0 periods of the same runs (separate events, not pre-cue).
    """
    from mne.datasets import eegbci

    runs = [3, 7, 11] if task == "execution" else [4, 8, 12]
    files = eegbci.load_data(subject, runs, path=str(EXT / "physionet"), update_path=False, verbose="ERROR")
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose="ERROR") for f in files], verbose="ERROR")
    eegbci.standardize(raw)
    raw = _finish(raw)
    ev, eid = mne.events_from_annotations(raw, event_id={"T0": 0, "T1": 1, "T2": 2}, verbose="ERROR")
    move = ev[ev[:, 2] > 0]
    rest = ev[ev[:, 2] == 0].copy()
    rest[:, 2] = 99
    return Recording(
        "physionet", f"S{subject:03d}", raw, move, {"left_hand": 1, "right_hand": 2},
        {"left_hand": "left", "right_hand": "right"}, task=task,
        move_win=(0.5, 2.0), rest_mode="events", rest_win=(0.5, 2.0), rest_events=rest,
        population="healthy", meta=dict(no_eog=True),
    )


# --------------------------------------------------------------------------- bnci2014
def load_bnci2014(subject: int) -> Recording:
    """BCI Competition IV-2a (BNCI 2014-001), session 1 ('0train'): left / right hand MI.
    Fixation cross at t=0, cue at t=2 s, imagery 3-6 s. Events are at CUE onset."""
    from moabb.datasets import BNCI2014_001

    for key in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH"):
        mne.set_config(key, str(EXT / "moabb"), set_env=True)
    data = BNCI2014_001().get_data(subjects=[subject])[subject]
    sess = data["0train"]
    raws = [sess[k] for k in sorted(sess)]
    raw = mne.concatenate_raws([r.copy() for r in raws], verbose="ERROR")
    ev, _ = mne.events_from_annotations(raw, event_id={"left_hand": 1, "right_hand": 2}, verbose="ERROR")
    raw.drop_channels([c for c in ("STI", "stim") if c in raw.ch_names])
    raw = _finish(raw, eog=["EOG1", "EOG2", "EOG3"])
    return Recording(
        "bnci2014", f"A{subject:02d}", raw, ev, {"left_hand": 1, "right_hand": 2},
        {"left_hand": "left", "right_hand": "right"}, task="imagery",
        move_win=(0.5, 2.0), rest_mode="precue", rest_win=(-1.5, 0.0), population="healthy",
    )


LOADERS = {
    "liu2024": (load_liu2024, [s for s in range(1, 51) if s not in (5, 15, 49)]),
    "bnci2019": (load_bnci2019, list(range(1, 10))),
    "physionet_exec": (lambda s: load_physionet(s, "execution"), [s for s in range(1, 51) if s not in (38, 88, 89, 92, 100, 104)]),
    "physionet_imag": (lambda s: load_physionet(s, "imagery"), [s for s in range(1, 51) if s not in (38, 88, 89, 92, 100, 104)]),
    "bnci2014": (load_bnci2014, list(range(1, 10))),
}
