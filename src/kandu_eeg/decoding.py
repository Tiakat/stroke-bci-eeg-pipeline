"""Decoding motor intent (affected-hand MI vs rest) - the IpsiHand problem.

Kandu/IpsiHand does not need left-vs-right classification: it needs to know
*when* the patient tries to move the affected hand (open the orthosis) vs rests.
So the main task is a binary "MI vs rest" detector built only from trials of the
affected hand.

Cross-validation design (leakage control)
-----------------------------------------
Each trial gives one REST window (instruction period) and one MI window. Both
windows of a trial always go to the same fold (StratifiedGroupKFold, group =
trial). Every data-driven choice (best frequency bin, CSP filters, channel
subset) is made INSIDE the training fold. Otherwise the AUC is optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from . import features as F
from . import spatial as S


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class IntentDataset:
    subject: int
    X: np.ndarray            # (n_windows, n_channels, n_times) - offline-filtered 1-40 Hz
    y: np.ndarray            # 1 = MI, 0 = rest
    groups: np.ndarray       # trial index
    ch_names: list
    sfreq: float
    hemi: str                # contralesional hemisphere ('left'/'right')
    hand: str                # which hand's trials ('affected'/'unaffected')

    def idx(self, names):
        return [self.ch_names.index(n) for n in names]


def build_intent_dataset(epochs, subject: int, hemi: str, hand_label: str, hand: str,
                         keep_mask: np.ndarray | None = None) -> IntentDataset:
    """Cut MI and REST windows (same length) from epochs of one hand."""
    sel = epochs.events[:, 2] == epochs.event_id[hand_label]
    if keep_mask is not None:
        sel &= keep_mask
    eeg = [c for c in epochs.ch_names if c not in C.EOG_CHANNELS]
    D = epochs.get_data(picks=eeg)[sel]
    t = epochs.times
    mi = (t >= C.MI_WIN[0]) & (t < C.MI_WIN[1])
    rs = (t >= C.REST_WIN[0]) & (t < C.REST_WIN[1])
    n = min(mi.sum(), rs.sum())
    Xmi, Xrs = D[..., np.flatnonzero(mi)[:n]], D[..., np.flatnonzero(rs)[:n]]
    trial_ids = np.flatnonzero(sel)
    X = np.concatenate([Xrs, Xmi])
    y = np.r_[np.zeros(len(Xrs), int), np.ones(len(Xmi), int)]
    g = np.r_[trial_ids, trial_ids]
    return IntentDataset(subject, X, y, g, eeg, epochs.info["sfreq"], hemi, hand)


# --------------------------------------------------------------------------- #
# Signal derivations for the control channel
# --------------------------------------------------------------------------- #
def control_signal(ds: IntentDataset, derivation: str, site: str = C.CONTROL_SITE) -> np.ndarray:
    """Return (n_windows, n_times) for one control channel under a given reference.

    derivation:
      'raw'      -> as recorded (reference = CPz)
      'car6'     -> CAR over the kandu6 montage (5 helpers + control)
      'car8'     -> CAR over the Bundy-2017 8-channel montage
      'car29'    -> CAR over the full cap
      'lap4'     -> small Laplacian: C - mean(FC, CP, T, Cz)
      'bip_cp'   -> C - CP (2 electrodes)
      'bip_fc'   -> C - FC (2 electrodes)
    """
    ch = S.resolve(site, ds.hemi)
    X = ds.X
    names = ds.ch_names
    if derivation == "raw":
        return X[:, names.index(ch)]
    if derivation.startswith("car"):
        pool = {"car6": S.montage_channels("kandu6", ds.hemi),
                "car8": S.montage_channels("bundy8", ds.hemi),
                "car29": names}[derivation]
        if ch not in pool:
            pool = pool + [ch]
        Xc = S.car(X, ds.idx(pool))
        return Xc[:, names.index(ch)]
    if derivation == "lap4":
        return S.laplacian(X, names, ch, S.laplacian_neighbours(ch))
    if derivation == "bip_cp":
        return S.bipolar(X, names, ch, S.resolve(site.replace(":C", ":CP"), ds.hemi))
    if derivation == "bip_fc":
        return S.bipolar(X, names, ch, S.resolve(site.replace(":C", ":FC"), ds.hemi))
    raise ValueError(derivation)


# --------------------------------------------------------------------------- #
# Feature extractors
# --------------------------------------------------------------------------- #
FREQS = np.arange(6, 31, 1.0)


def spectrum_1hz(x: np.ndarray, sfreq: float, method: str = "welch") -> np.ndarray:
    """log10 power in 1-Hz bins 6-30 Hz. x: (n, n_times) -> (n, 25)."""
    if method == "ar":
        _, p = F.ar_psd(x, sfreq, order=16, freqs=FREQS)
    else:
        f, pw = signal.welch(x, fs=sfreq, nperseg=min(x.shape[-1], int(sfreq)), axis=-1)
        p = np.stack([np.interp(FREQS, f, pp) for pp in pw])
    return np.log10(p)


def bandpower_feats(X: np.ndarray, sfreq: float, bands=("mu", "low_beta", "high_beta")) -> np.ndarray:
    """(n, ch, t) -> (n, ch*len(bands)) log band power."""
    bp = F.band_log_power(X, sfreq, {b: C.BANDS[b] for b in bands})
    return bp.reshape(len(X), -1)


# --------------------------------------------------------------------------- #
# Cross-validation engine
# --------------------------------------------------------------------------- #
def _cv_splits(y, groups, n_splits=5, n_repeats=5, seed=C.RANDOM_STATE):
    n_splits = min(n_splits, len(np.unique(groups)) // 2)
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        yield from cv.split(np.zeros(len(y)), y, groups)


def cv_auc(fit_predict, n_samples_y, y, groups, n_splits=5, n_repeats=5, seed=C.RANDOM_STATE):
    """Generic CV: fit_predict(train_idx, test_idx) -> decision scores on test. Returns mean AUC."""
    aucs = []
    for tr, te in _cv_splits(y, groups, n_splits, n_repeats, seed):
        s = fit_predict(tr, te)
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs))


def best_bin_decoder(feat: np.ndarray, y: np.ndarray, erd_only: bool = True):
    """Bundy-2017 style: pick the single 1-Hz bin with the strongest effect on the
    TRAINING data, then LDA on that one feature.

    erd_only=True: only power DECREASES (negative r²) are eligible - the
    physiological prior used by IpsiHand. With ~20 trials, an unconstrained |r²|
    search often locks onto a spurious power increase that does not generalise.
    """
    def fp(tr, te):
        r2 = F.signed_r2(feat[tr][y[tr] == 1], feat[tr][y[tr] == 0])
        k = int(np.argmin(r2)) if erd_only else int(np.argmax(np.abs(r2)))
        clf = LDA().fit(feat[tr][:, [k]], y[tr])
        return clf.decision_function(feat[te][:, [k]])
    return fp


def lda_decoder(feat: np.ndarray, y: np.ndarray, shrink: bool = True):
    def fp(tr, te):
        clf = LDA(solver="lsqr", shrinkage="auto") if shrink else LDA()
        clf.fit(feat[tr], y[tr])
        return clf.decision_function(feat[te])
    return fp


def csp_decoder(Xb: np.ndarray, y: np.ndarray, n_components: int = 4):
    from mne.decoding import CSP

    def fp(tr, te):
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
        f_tr = csp.fit_transform(Xb[tr], y[tr])
        clf = LDA().fit(f_tr, y[tr])
        return clf.decision_function(csp.transform(Xb[te]))
    return fp


def tangent_space_decoder(Xb: np.ndarray, y: np.ndarray):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression

    covs = Covariances("oas").fit_transform(Xb)

    def fp(tr, te):
        ts = TangentSpace(metric="riemann").fit(covs[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(ts.transform(covs[tr]), y[tr])
        return clf.decision_function(ts.transform(covs[te]))
    return fp


def bandpass(X: np.ndarray, sfreq: float, lo: float = 8.0, hi: float = 30.0) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1, padlen=min(X.shape[-1] - 1, 3 * 13))


# --------------------------------------------------------------------------- #
# Greedy channel selection (nested inside CV)
# --------------------------------------------------------------------------- #
def greedy_selection_curve(Xfeat_ch: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           k_max: int = 6, n_splits=5, n_repeats=3):
    """AUC as a function of the number of channels, channels chosen greedily on the
    TRAINING folds only (inner CV), evaluated on the outer test fold.

    Xfeat_ch: (n, n_channels, n_feats_per_channel)
    Returns auc[k-1] for k = 1..k_max, and the selection counts per channel.
    """
    n, n_ch, _ = Xfeat_ch.shape
    auc = {k: [] for k in range(1, k_max + 1)}
    counts = np.zeros((k_max, n_ch))
    for tr, te in _cv_splits(y, groups, n_splits, n_repeats):
        chosen = []
        for k in range(1, k_max + 1):
            best, best_c = -1, None
            for c in range(n_ch):
                if c in chosen:
                    continue
                cand = chosen + [c]
                ftr = Xfeat_ch[tr][:, cand].reshape(len(tr), -1)
                inner = cv_auc(lda_decoder(ftr, y[tr]), None, y[tr], groups[tr], n_splits=3, n_repeats=1)
                if inner > best:
                    best, best_c = inner, c
            chosen.append(best_c)
            counts[k - 1, best_c] += 1
            ftr = Xfeat_ch[tr][:, chosen].reshape(len(tr), -1)
            fte = Xfeat_ch[te][:, chosen].reshape(len(te), -1)
            clf = LDA(solver="lsqr", shrinkage="auto").fit(ftr, y[tr])
            if len(np.unique(y[te])) == 2:
                auc[k].append(roc_auc_score(y[te], clf.decision_function(fte)))
    return np.array([np.mean(auc[k]) for k in range(1, k_max + 1)]), counts


def permutation_pvalue(fit_predict_factory, feat, y, groups, observed: float, n_perm: int = 200,
                       seed: int = 0) -> float:
    """Shuffle labels *within trial pairs* (keeps the group structure) and recompute AUC."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            if rng.random() < 0.5:
                yp[idx] = yp[idx][::-1]
        null.append(cv_auc(fit_predict_factory(feat, yp), None, yp, groups, n_repeats=1))
    null = np.array(null)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))
