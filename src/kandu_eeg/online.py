"""Pseudo-online simulation of an IpsiHand-style controller.

Offline analysis can cheat in ways a device cannot: zero-phase filters look into
the future, epochs are cut with knowledge of the cue, features are normalised
with statistics of the whole session. Here every step is causal:

    causal band-pass (Butterworth, sosfilt with carried state)
 -> CAR over the 6 montage electrodes
 -> sliding 500-ms window every 125 ms on the control channel   (Bundy et al. 2017)
 -> AR (maximum-entropy) power at the patient's control frequency
 -> z-score w.r.t. REST statistics from calibration
 -> integrate into a hand-aperture command Y(t) in [0, 100] %:
        Y(t) = Y(t-1) + gain * [(x(t) - mu_rest - bias) / sd_rest] * sign(mu_move - mu_rest)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import config as C
from . import features as F
from . import spatial as S
from .io import mi_events, segments


@dataclass
class Calibration:
    freq: float
    mu_rest: float
    sd_rest: float
    mu_move: float
    r2: float


def causal_preprocess(raw, sfreq_out: float = C.SFREQ_TARGET, l_freq=1.0, h_freq=40.0, order=4):
    """Causal IIR band-pass per segment (state reset at each segment start), then decimate."""
    sf = raw.info["sfreq"]
    sos = signal.butter(order, [l_freq, h_freq], btype="bandpass", fs=sf, output="sos")
    X = raw.get_data(picks="eeg").copy()
    for a, b in segments(raw):
        seg = X[:, a:b] - X[:, a:a + 1]          # remove the offset with the FIRST sample only (causal)
        zi = signal.sosfilt_zi(sos)[:, None, :] * seg[:, :1][None]
        X[:, a:b], _ = signal.sosfilt(sos, seg, axis=-1, zi=zi)
    step = int(round(sf / sfreq_out))
    return X[:, ::step], sf / step                # h_freq=40 < new Nyquist (125) -> no aliasing


def sliding_power(x: np.ndarray, sfreq: float, freqs: np.ndarray, win_s=0.5, step_s=0.125, order=16):
    """AR power for each window ending at sample i (causal). Returns (times_idx, P[n_win, n_freq])."""
    w, st = int(win_s * sfreq), int(step_s * sfreq)
    ends = np.arange(w, len(x) + 1, st)
    segs = np.stack([x[e - w:e] for e in ends])
    _, P = F.ar_psd(segs, sfreq, order=order, freqs=freqs)
    return ends - 1, np.log10(P)


def run_subject(raw, hemi: str, affected_label: str, freqs=np.arange(6, 31, 1.0), gain=0.08, bias=0.5):
    """Calibrate on the first half of affected-hand trials, test on the second half."""
    X, sf = causal_preprocess(raw)
    names = [c for c in raw.ch_names if c not in C.EOG_CHANNELS]
    pool = [names.index(c) for c in S.montage_channels("kandu6", hemi)]
    ctrl = names.index(S.resolve(C.CONTROL_SITE, hemi))
    x = (X - X[pool].mean(0, keepdims=True))[ctrl]            # CAR6, control channel

    ends, P = sliding_power(x, sf, freqs)
    t_end = ends / sf
    events, ev_id = mi_events(raw)
    ev_t = events[:, 0] / raw.info["sfreq"]
    aff = events[:, 2] == ev_id[affected_label]
    aff_t = ev_t[aff]
    half = len(aff_t) // 2
    calib_t, test_t = aff_t[:half], aff_t[half:]

    def windows_in(onsets, lo, hi):
        m = np.zeros(len(t_end), bool)
        for o in onsets:
            m |= (t_end >= o + lo) & (t_end < o + hi)
        return m

    # calibration: windows fully inside REST (instruction) or MI periods
    rest_m = windows_in(calib_t, C.REST_WIN[0] + 0.5, C.REST_WIN[1])
    mi_m = windows_in(calib_t, C.MI_WIN[0] + 0.5, C.MI_WIN[1] + 1.0)
    r2 = F.signed_r2(P[mi_m], P[rest_m])
    k = int(np.argmin(r2))                                    # most negative = strongest ERD
    cal = Calibration(float(freqs[k]), P[rest_m, k].mean(), P[rest_m, k].std(), P[mi_m, k].mean(), float(r2[k]))

    # run the controller over the test trials
    feat = P[:, k]
    Y = np.zeros(len(feat))
    sgn = np.sign(cal.mu_move - cal.mu_rest)
    for i in range(1, len(feat)):
        z = (feat[i] - cal.mu_rest) / cal.sd_rest
        # the device only *opens* the hand on desynchronisation: sgn*z > bias
        Y[i] = np.clip(Y[i - 1] + gain * 100 * (sgn * z - bias) / 4.0, 0, 100)
    # per-trial traces aligned to MI onset (test half)
    lags = np.arange(-1.5, 4.0 + 1e-9, 0.125)
    traces = []
    for o in test_t:
        idx = [np.argmin(np.abs(t_end - (o + l))) for l in lags]
        traces.append(Y[idx])
    rest_mt = windows_in(test_t, C.REST_WIN[0] + 0.5, C.REST_WIN[1])
    mi_mt = windows_in(test_t, C.MI_WIN[0] + 0.5, C.MI_WIN[1] + 1.0)
    from sklearn.metrics import roc_auc_score

    yy = np.r_[np.zeros(rest_mt.sum()), np.ones(mi_mt.sum())]
    ss = np.r_[sgn * feat[rest_mt], sgn * feat[mi_mt]]          # higher = more "move-like"
    return dict(cal=cal, t=t_end, feat=feat, Y=Y, lags=lags, traces=np.array(traces),
                auc_test_windows=float(roc_auc_score(yy, ss)), test_onsets=test_t)
