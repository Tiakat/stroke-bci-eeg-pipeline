"""One analysis, every dataset: the cross-dataset replication engine.

For each recording and each hand we ask the Kandu questions with IDENTICAL code:

1. Can MOVE (attempt / imagery / execution) be told apart from REST?
   - with ONE electrode, ipsilateral or contralateral to the hand, under three references
   - with a 6-electrode one-hemisphere montage (Kandu-like) used as features
   - with all electrodes (upper bound)
   - with EOG / frontal-pole channels only (confound baseline)
   - with slow movement-related cortical potentials (MRCP, 0.3-3 Hz)
2. How lateralised is the ERD (contralateral vs ipsilateral C electrode)?
3. Left vs right hand (paradigm confounds matched): only for datasets with both hands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

import mne

from . import config as C
from . import decoding as D
from . import features as F
from . import preprocessing as P
from . import spatial as S
from .datasets import Recording

# --------------------------------------------------------------------------- #
# Electrode resolution with fallbacks (caps differ between datasets)
# --------------------------------------------------------------------------- #
SITES = {   # hemisphere-generic site -> candidate names (left, right), first available wins
    "F": [("F3", "F4"), ("FFC3h", "FFC4h"), ("FC1", "FC2")],
    "FC": [("FC3", "FC4")],
    "C": [("C3", "C4")],
    "CP": [("CP3", "CP4"), ("CCP3h", "CCP4h")],
    "P": [("P3", "P4"), ("CP1", "CP2")],
    "LAT": [("T7", "T8"), ("C5", "C6"), ("FC5", "FC6")],
}
EOG_PROXY = ["Fp1", "Fp2", "AF7", "AF8", "AFz", "F7", "F8"]
MRCP_CHANNELS = ["FC1", "FCz", "FC2", "C1", "Cz", "C2", "C3", "C4", "CP1", "CPz", "CP2"]


def site(name: str, side: str, chs: list[str]) -> str | None:
    for left, right in SITES[name]:
        c = left if side == "left" else right
        if c in chs:
            return c
    return None


def opposite(side):
    return "right" if side == "left" else "left"


def montage6(side: str, chs: list[str]) -> list[str]:
    out = [site(s, side, chs) for s in ["F", "FC", "C", "CP", "P"]] + (["Cz"] if "Cz" in chs else [])
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# Pre-processing (same chain as the stroke pipeline)
# --------------------------------------------------------------------------- #
LINE = {"liu2024": 50.0, "bnci2019": 50.0, "physionet": 60.0, "bnci2014": 50.0}


def preprocess(rec: Recording, l_freq=1.0, h_freq=40.0, eog=True):
    raw = P.remove_dc(rec.raw)
    f0 = LINE[rec.dataset]
    notch = tuple(f for f in (f0, 2 * f0, 3 * f0, 4 * f0) if f < raw.info["sfreq"] / 2)
    raw = P.filter_offline(raw, l_freq=l_freq, h_freq=h_freq, notch=notch)
    qc = P.channel_qc(raw)
    slope = P.aperiodic_slope(raw)
    raw.info["bads"] = list(qc.index[qc["bad"]])
    if raw.info["sfreq"] > 260:
        raw = P.downsample(raw, 250.0)
    if eog and len(mne.pick_types(raw.info, eog=True)):
        raw, _ = P.eog_regression(raw)
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw, dict(n_bad=len(qc.index[qc["bad"]]), slope=slope)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def rescale_events(events: np.ndarray, sf_from: float, sf_to: float) -> np.ndarray:
    """Event sample indices are tied to a sampling rate: after resampling 500 -> 250 Hz,
    sample 1000 becomes sample 500. Forgetting this shifts every epoch (a silent bug)."""
    ev = events.copy()
    ev[:, 0] = np.round(ev[:, 0] * sf_to / sf_from).astype(int)
    return ev


def _epochs(raw, events, tmin, tmax, picks):
    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True,
                      picks=picks, reject_by_annotation=True, verbose="ERROR", event_repeated="drop")


def windows(rec: Recording, raw, label: str, rest_source: str = "auto", picks=("eeg", "eog"),
            ptp=150e-6, roi=None):
    """Return X (n, ch, t), y (1=move), groups, ch_names for one hand."""
    code = rec.event_id[label]
    sf = raw.info["sfreq"]
    ev = rescale_events(rec.events[rec.events[:, 2] == code], rec.sfreq, sf)
    L = rec.move_win[1] - rec.move_win[0]
    n = int(round(L * sf))
    picks = list(picks)
    epm = _epochs(raw, ev, rec.move_win[0], rec.move_win[1], picks)
    Xm = epm.get_data()[..., :n]
    names = epm.ch_names
    use_events_rest = (rest_source == "events") or (rest_source == "auto" and rec.rest_mode == "events")
    if use_events_rest:
        rv = rescale_events(rec.rest_events, rec.sfreq, sf)
        rv = rv[rv[:, 0] + int(rec.rest_win[1] * sf) < raw.n_times]
        epr = _epochs(raw, rv, rec.rest_win[0], rec.rest_win[1], picks)
        Xr = epr.get_data()[..., :n]
        gm = np.arange(len(Xm))
        gr = 10_000 + np.arange(len(Xr))
    else:
        epr = _epochs(raw, ev, rec.rest_win[0], rec.rest_win[1], picks)
        Xr = epr.get_data()[..., :n]
        # keep only trials present in both
        sel_m = epm.selection
        sel_r = epr.selection
        common = np.intersect1d(sel_m, sel_r)
        Xm = Xm[np.isin(sel_m, common)]
        Xr = Xr[np.isin(sel_r, common)]
        gm = gr = np.arange(len(common))
    X = np.concatenate([Xr, Xm])
    y = np.r_[np.zeros(len(Xr), int), np.ones(len(Xm), int)]
    g = np.r_[gr, gm]
    if ptp is not None and roi:
        idx = [names.index(c) for c in roi if c in names]
        amp = X[:, idx].max(-1) - X[:, idx].min(-1)          # (n, roi) peak-to-peak
        # adaptive threshold: amplitudes differ between amplifiers / references, so a
        # fixed 150 µV can reject everything (PhysioNet). Use max(150 µV, 3 x median p-p).
        thr = max(ptp, 3 * np.median(amp))
        ok = (amp < thr).all(1)
        if not use_events_rest:   # drop pairs together
            bad_groups = np.unique(g[~ok])
            ok = ~np.isin(g, bad_groups)
        X, y, g = X[ok], y[ok], g[ok]
    return X, y, g, names


# --------------------------------------------------------------------------- #
# Feature helpers
# --------------------------------------------------------------------------- #
def _bestbin_auc(x, y, g, sf):
    return D.cv_auc(D.best_bin_decoder(D.spectrum_1hz(x, sf), y), None, y, g)


def _bp_auc(X, y, g, sf):
    return D.cv_auc(D.lda_decoder(D.bandpower_feats(X, sf), y), None, y, g)


def _ts_auc(X, y, g, sf):
    return D.cv_auc(D.tangent_space_decoder(D.bandpass(X, sf), y), None, y, g)


def _eog_auc(X, y, g, sf):
    f = F.band_log_power(X, sf, {"slow": (1, 4), "mu": (8, 13), "lb": (13, 20), "hb": (20, 30)})
    return D.cv_auc(D.lda_decoder(f.reshape(len(X), -1), y), None, y, g)


# --------------------------------------------------------------------------- #
# Main per-recording analysis
# --------------------------------------------------------------------------- #
def analyse(rec: Recording, labels: list[str] | None = None, mrcp: bool = True) -> dict:
    raw, q = preprocess(rec)
    sf = raw.info["sfreq"]
    eeg_names = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    eog_names = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eog"]
    labels = labels or list(rec.event_id)
    rows, erd_rows = [], []
    base = dict(dataset=rec.dataset, subject=rec.subject, task=rec.task, population=rec.population)

    for lab in labels:
        side = rec.hand[lab]                          # moving hand
        contra, ipsi = opposite(side), side           # hemispheres
        roi = [c for c in [site("FC", s, eeg_names) for s in ("left", "right")] +
               [site("C", s, eeg_names) for s in ("left", "right")] + ["Cz"] if c]
        X, y, g, names = windows(rec, raw, lab, roi=roi)
        if len(np.unique(g[y == 1])) < 8:
            continue
        role = lab
        if rec.dataset == "liu2024":
            role = "affected" if side == rec.meta["affected_hand"] else "unaffected"
        hb = dict(base, hand=side, role=role, n_move=int((y == 1).sum()), n_rest=int((y == 0).sum()))
        ix = {c: i for i, c in enumerate(names)}
        Xe = X[:, [ix[c] for c in eeg_names]]
        Xcar = S.car(Xe)

        for hemi_role, hemi in (("ipsi", ipsi), ("contra", contra)):
            c_ = site("C", hemi, eeg_names)
            fc_ = site("FC", hemi, eeg_names)
            m6 = montage6(hemi, eeg_names)
            ci = eeg_names.index(c_)
            # one electrode, three references
            x_bip = Xe[:, ci] - Xe[:, eeg_names.index(fc_)]
            Xm6 = S.car(Xe[:, [eeg_names.index(c) for c in m6]])
            x_car6 = Xm6[:, m6.index(c_)]
            nb = [n for n in [fc_, site("CP", hemi, eeg_names), site("LAT", hemi, eeg_names), "Cz"] if n and n in eeg_names]
            x_lap = Xe[:, ci] - Xe[:, [eeg_names.index(n) for n in nb]].mean(1)
            for pipe, x in (("1ch_bipolar", x_bip), ("1ch_car6", x_car6), ("1ch_laplacian", x_lap), ("1ch_car_all", Xcar[:, ci])):
                rows.append(dict(hb, pipeline=pipe, hemisphere=hemi_role, n_el={"1ch_bipolar": 2, "1ch_car6": 6, "1ch_laplacian": len(nb) + 1, "1ch_car_all": len(eeg_names)}[pipe],
                                 auc=_bestbin_auc(x, y, g, sf)))
            rows.append(dict(hb, pipeline="6ch_bandpower", hemisphere=hemi_role, n_el=len(m6), auc=_bp_auc(Xm6, y, g, sf)))
            rows.append(dict(hb, pipeline="6ch_tangent", hemisphere=hemi_role, n_el=len(m6), auc=_ts_auc(Xm6, y, g, sf)))
            # ERD (dB, mu+beta) at that C electrode, CAR over all
            pm = F.band_log_power(Xcar[y == 1][:, [ci]], sf, {"mb": (8, 30)})[..., 0, 0]
            pr = F.band_log_power(Xcar[y == 0][:, [ci]], sf, {"mb": (8, 30)})[..., 0, 0]
            erd_rows.append(dict(hb, hemisphere=hemi_role, electrode=c_, erd_db=10 * (pm.mean() - pr.mean())))

        rows.append(dict(hb, pipeline="all_tangent", hemisphere="both", n_el=len(eeg_names), auc=_ts_auc(Xe, y, g, sf)))
        if eog_names:
            Xo = X[:, [ix[c] for c in eog_names]]
            rows.append(dict(hb, pipeline="eog_only", hemisphere="none", n_el=len(eog_names), auc=_eog_auc(Xo, y, g, sf)))
        prox = [c for c in EOG_PROXY if c in eeg_names][:4]
        if prox:
            rows.append(dict(hb, pipeline="frontal_pole", hemisphere="none", n_el=len(prox),
                             auc=_eog_auc(Xe[:, [eeg_names.index(c) for c in prox]], y, g, sf)))
        sm = [site(s, h, eeg_names) for h in ("left", "right") for s in ("FC", "C", "CP")]
        rows.append(dict(hb, pipeline="sensorimotor6_bandpower", hemisphere="both", n_el=6,
                         auc=_bp_auc(S.car(Xe)[:, [eeg_names.index(c) for c in sm]], y, g, sf)))

    out = dict(rows=rows, erd=erd_rows, qc=dict(base, **q))

    # ---------------- left vs right hand, MOVE window only (confounds matched)
    if len(rec.event_id) == 2:
        evs, Xs, ys = [], [], []
        labs = list(rec.event_id)
        for k, lab in enumerate(labs):
            ev = rescale_events(rec.events[rec.events[:, 2] == rec.event_id[lab]], rec.sfreq, sf)
            ep = _epochs(raw, ev, rec.move_win[0], rec.move_win[1], ["eeg", "eog"])
            n = int(round((rec.move_win[1] - rec.move_win[0]) * sf))
            Xs.append(ep.get_data()[..., :n]); ys.append(np.full(len(ep), k))
        Xh, yh = np.concatenate(Xs), np.concatenate(ys)
        gh = np.arange(len(yh))
        nm = ep.ch_names
        Xe = Xh[:, [nm.index(c) for c in eeg_names]]
        sm = [site(s, h, eeg_names) for h in ("left", "right") for s in ("FC", "C", "CP")]
        hv = dict(base, hand="both", role="left_vs_right", n_move=len(yh), n_rest=0)
        out["rows"] += [
            dict(hv, pipeline="lr_sensorimotor6", hemisphere="both", n_el=6, auc=_bp_auc(S.car(Xe)[:, [eeg_names.index(c) for c in sm]], yh, gh, sf)),
            dict(hv, pipeline="lr_all_tangent", hemisphere="both", n_el=len(eeg_names), auc=_ts_auc(Xe, yh, gh, sf)),
        ]
        if eog_names:
            out["rows"].append(dict(hv, pipeline="lr_eog_only", hemisphere="none", n_el=len(eog_names),
                                    auc=_eog_auc(Xh[:, [nm.index(c) for c in eog_names]], yh, gh, sf)))

    # ---------------- MRCP: slow potentials 0.3-3 Hz, time-domain features
    if mrcp:
        out["rows"] += mrcp_rows(rec, base)
    # ---------------- true rest (BNCI2019 has dedicated rest runs)
    if rec.dataset == "bnci2019" and rec.rest_events is not None:
        lab = labels[0]
        X, y, g, names = windows(rec, raw, lab, rest_source="events", roi=None)
        Xe = X[:, [names.index(c) for c in eeg_names]]
        hb = dict(base, hand=rec.hand[lab], role="vs_true_rest", n_move=int((y == 1).sum()), n_rest=int((y == 0).sum()))
        hemi = rec.hand[lab]
        m6 = montage6(hemi, eeg_names)
        out["rows"] += [
            dict(hb, pipeline="6ch_bandpower", hemisphere="ipsi", n_el=6, auc=_bp_auc(S.car(Xe[:, [eeg_names.index(c) for c in m6]]), y, g, sf)),
            dict(hb, pipeline="all_tangent", hemisphere="both", n_el=len(eeg_names), auc=_ts_auc(Xe, y, g, sf)),
            dict(hb, pipeline="eog_only", hemisphere="none", n_el=len(eog_names), auc=_eog_auc(X[:, [names.index(c) for c in eog_names]], y, g, sf)),
        ]
    return out


def mrcp_rows(rec: Recording, base: dict) -> list[dict]:
    """Movement-related cortical potentials: a slow negative shift over the motor strip
    that starts before / at movement onset. Kept in the 0.3-3 Hz band, decimated to 10 Hz,
    the time course itself is the feature (Ofner et al. 2019 used exactly this)."""
    raw = P.remove_dc(rec.raw)
    raw = raw.copy().filter(0.3, 3.0, method="iir", iir_params=dict(order=4, ftype="butter"),
                            phase="zero", picks=["eeg", "eog"], verbose="ERROR")
    if raw.info["sfreq"] > 260:
        raw = P.downsample(raw, 250.0)
    if len(mne.pick_types(raw.info, eog=True)):
        raw, _ = P.eog_regression(raw)
    chs = [c for c in MRCP_CHANNELS if c in raw.ch_names]
    rows = []
    for lab in rec.event_id:
        X, y, g, names = windows(rec, raw, lab, picks=["eeg", "eog"], ptp=None)
        Xs = X[:, [names.index(c) for c in chs]]
        Xs = Xs - Xs[..., :1]                                # anchor at window start
        dec = signal.resample_poly(Xs, 10, int(raw.info["sfreq"]), axis=-1)  # -> 10 Hz
        f = dec.reshape(len(dec), -1)
        role = lab if rec.dataset != "liu2024" else ("affected" if rec.hand[lab] == rec.meta["affected_hand"] else "unaffected")
        rows.append(dict(base, hand=rec.hand[lab], role=role, pipeline="mrcp_0.3-3Hz", hemisphere="both",
                         n_el=len(chs), n_move=int((y == 1).sum()), n_rest=int((y == 0).sum()),
                         auc=D.cv_auc(D.lda_decoder(f, y), None, y, g)))
    return rows
