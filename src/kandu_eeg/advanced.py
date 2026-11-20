"""Advanced EEG analyses beyond the core decoding pipeline.

1. Spectral parameterisation (FOOOF): aperiodic exponent + oscillatory peaks
2. Stroke EEG markers: pairwise-derived Brain Symmetry Index (pdBSI), delta/alpha ratio (DAR)
3. Functional connectivity: weighted phase-lag index (wPLI)
4. Group statistics: cluster-based permutation test on time-frequency maps
5. Movement-related cortical potentials (MRCP) grand averages
6. Cross-subject transfer: Riemannian re-centring (calibration-free decoding)
7. Learning curves: how many calibration trials are needed
8. Deep learning: a compact EEGNet (Lawhern et al. 2018) vs Riemannian baseline
9. Reliability: split-half reliability of the ERD
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats

from . import config as C

# --------------------------------------------------------------------------- #
# 1. Spectral parameterisation
# --------------------------------------------------------------------------- #
def fooof_params(freqs: np.ndarray, psd: np.ndarray, fmin=2.0, fmax=40.0):
    """Fit FOOOF (Donoghue et al. 2020) to one spectrum.

    The spectrum is modelled as  log10 P(f) = b - log10(f^chi) + sum of Gaussian peaks.
      b   = offset (broadband power)
      chi = aperiodic exponent (steeper = more 'inhibition' / slower activity)
    Returns exponent, offset, alpha/mu peak frequency and its power above the 1/f.
    """
    from fooof import FOOOF

    fm = FOOOF(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.05, aperiodic_mode="fixed", verbose=False)
    fm.fit(freqs, psd, [fmin, fmax])
    off, exp = fm.aperiodic_params_
    peaks = fm.peak_params_  # (n, 3): CF, PW, BW
    alpha = peaks[(peaks[:, 0] >= 6) & (peaks[:, 0] <= 14)] if len(peaks) else np.empty((0, 3))
    beta = peaks[(peaks[:, 0] > 14) & (peaks[:, 0] <= 30)] if len(peaks) else np.empty((0, 3))
    a = alpha[np.argmax(alpha[:, 1])] if len(alpha) else (np.nan, np.nan, np.nan)
    b_ = beta[np.argmax(beta[:, 1])] if len(beta) else (np.nan, np.nan, np.nan)
    return dict(exponent=exp, offset=off, alpha_cf=a[0], alpha_pw=a[1], beta_cf=b_[0], beta_pw=b_[1],
                r2=fm.r_squared_, error=fm.error_)


# --------------------------------------------------------------------------- #
# 2. Stroke markers
# --------------------------------------------------------------------------- #
def pdbsi(freqs, psd_left, psd_right, fmin=1.0, fmax=25.0):
    """Pairwise-derived Brain Symmetry Index (van Putten 2007; Sheorajpanday et al. 2009).

    For each homologous pair and frequency: |R - L| / (R + L); averaged over pairs and
    frequencies. 0 = perfectly symmetric, 1 = maximal asymmetry. A clinical marker of
    stroke severity on resting EEG.
    psd_left/psd_right: (n_pairs, n_freqs) power.
    """
    m = (freqs >= fmin) & (freqs <= fmax)
    L, R = psd_left[:, m], psd_right[:, m]
    return float(np.mean(np.abs(R - L) / (R + L)))


def delta_alpha_ratio(freqs, psd):
    """Delta (1-4 Hz) / alpha (8-13 Hz) power ratio: higher = more slowing.
    Elevated DAR after stroke is linked to worse outcome (Finnigan et al. 2016)."""
    d = psd[..., (freqs >= 1) & (freqs < 4)].mean(-1)
    a = psd[..., (freqs >= 8) & (freqs < 13)].mean(-1)
    return d / a


# --------------------------------------------------------------------------- #
# 3. Connectivity
# --------------------------------------------------------------------------- #
def wpli_matrix(X: np.ndarray, sfreq: float, fmin: float, fmax: float):
    """Weighted phase-lag index over trials (Vinck et al. 2011).

    X: (n_trials, n_ch, n_times). wPLI uses only the IMAGINARY part of the
    cross-spectrum, so zero-lag coupling caused by volume conduction or a shared
    reference does not count as connectivity. Returns (n_ch, n_ch) in [0, 1].
    """
    from mne_connectivity import spectral_connectivity_epochs

    con = spectral_connectivity_epochs(X, method="wpli", mode="multitaper", sfreq=sfreq,
                                       fmin=fmin, fmax=fmax, faverage=True, verbose="ERROR")
    M = con.get_data(output="dense")[..., 0]
    return M + M.T


# --------------------------------------------------------------------------- #
# 4. Time-frequency maps + group cluster test
# --------------------------------------------------------------------------- #
def tfr_logratio(epochs_data, sfreq, times, freqs, ref_win, n_cycles=None):
    """Morlet power, averaged over trials, as 10*log10(power / mean power in ref_win).
    epochs_data: (n_trials, n_ch, n_times) -> (n_ch, n_freqs, n_times) in dB."""
    from mne.time_frequency import tfr_array_morlet

    n_cycles = freqs / 2.0 if n_cycles is None else n_cycles
    p = tfr_array_morlet(epochs_data, sfreq, freqs, n_cycles=n_cycles, output="power", verbose="ERROR").mean(0)
    m = (times >= ref_win[0]) & (times < ref_win[1])
    return 10 * np.log10(p / p[..., m].mean(-1, keepdims=True))


def cluster_test_1samp(data, n_permutations=1000, seed=0):
    """Group-level test of (subjects, freqs, times) maps against 0 with cluster-based
    permutation (Maris & Oostenveld 2007). Controls the family-wise error over ALL
    time-frequency points without Bonferroni's loss of power: neighbouring points are
    grouped into clusters, and only the cluster mass is tested against a sign-flip null."""
    from mne.stats import permutation_cluster_1samp_test

    t_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(
        data, n_permutations=n_permutations, tail=0, seed=seed, out_type="mask", verbose="ERROR")
    sig = np.zeros(data.shape[1:], bool)
    for c, p in zip(clusters, p_vals):
        if p < 0.05:
            sig |= c
    return t_obs, sig, p_vals


# --------------------------------------------------------------------------- #
# 6. Cross-subject transfer
# --------------------------------------------------------------------------- #
def recentre(covs: np.ndarray) -> np.ndarray:
    """Riemannian re-centring (Zanini et al. 2018): whiten every covariance of ONE
    subject by that subject's own Riemannian mean, M^-1/2 C M^-1/2. Every subject's data
    then sit around the identity, which removes most between-subject (and between-session)
    shift. Uses no labels -> legitimate for a calibration-free BCI."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann

    M = mean_riemann(covs)
    W = invsqrtm(M)
    return np.array([W @ c @ W for c in covs])


def loso_tangent(cov_by_subj: dict, y_by_subj: dict, recentre_flag=True, metric=None):
    """Leave-one-subject-out: train tangent space + logistic regression on all other
    subjects, test on the held-out one. Returns {subject: AUC}.

    Tangent point = log-Euclidean mean (closed form, expm(mean(logm(C))), fast on
    thousands of pooled 64x64 matrices); mapping = affine-invariant Riemannian log map,
    which is scale-free. Covariances are expressed in µV²: in V² (~1e-11) a
    scale-dependent mapping gives features so small that the regularised logistic
    regression shrinks every weight to zero (constant scores, AUC exactly 0.5)."""
    metric = metric or {"mean": "logeuclid", "map": "riemann"}
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    subs = list(cov_by_subj)
    covs = {s: (recentre(cov_by_subj[s]) if recentre_flag else cov_by_subj[s] * 1e12) for s in subs}
    out = {}
    for s in subs:
        tr = [k for k in subs if k != s]
        Xtr = np.concatenate([covs[k] for k in tr]); ytr = np.concatenate([y_by_subj[k] for k in tr])
        ts = TangentSpace(metric=metric).fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(ts.transform(Xtr), ytr)
        out[s] = roc_auc_score(y_by_subj[s], clf.decision_function(ts.transform(covs[s])))
    return out


# --------------------------------------------------------------------------- #
# 7. Learning curve
# --------------------------------------------------------------------------- #
def learning_curve(covs, y, groups, sizes, n_rep=20, seed=0):
    """Within-subject AUC as a function of the number of TRAINING trials per class.
    Test set = a fixed random 30% of trials (pairs kept together), repeated n_rep times."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    res = {k: [] for k in sizes}
    for _ in range(n_rep):
        test_g = rng.choice(ug, size=max(2, int(0.3 * len(ug))), replace=False)
        te = np.isin(groups, test_g)
        pool = np.flatnonzero(~te)
        if len(np.unique(y[te])) < 2:
            continue
        for k in sizes:
            idx = np.r_[rng.permutation(pool[y[pool] == 0])[:k], rng.permutation(pool[y[pool] == 1])[:k]]
            if min((y[idx] == 0).sum(), (y[idx] == 1).sum()) < k:
                continue
            ts = TangentSpace(metric="riemann").fit(covs[idx])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(ts.transform(covs[idx]), y[idx])
            res[k].append(roc_auc_score(y[te], clf.decision_function(ts.transform(covs[te]))))
    return {k: (np.mean(v) if v else np.nan) for k, v in res.items()}


# --------------------------------------------------------------------------- #
# 8. EEGNet
# --------------------------------------------------------------------------- #
def make_eegnet(n_ch, n_times, F1=8, D=2, F2=16, kern=64, drop=0.5):
    """EEGNet-8,2 (Lawhern et al. 2018), ~3k parameters.

    Block 1: temporal conv (learns band-pass filters, kernel = 0.25 s at 250 Hz)
             -> depthwise spatial conv (D spatial filters per temporal filter = learned CSP)
             -> BatchNorm, ELU, average-pool, dropout
    Block 2: separable conv (summarises each feature map over time) -> pool -> dropout
    Head:    flatten -> linear -> 1 logit (binary MOVE vs REST)
    """
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False), nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.b2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(drop))
            import torch
            with torch.no_grad():
                n = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_times))).numel()
            self.head = nn.Linear(n, 1)

        def forward(self, x):
            return self.head(self.b2(self.b1(x)).flatten(1)).squeeze(1)

    return EEGNet()


def train_eegnet(Xtr, ytr, Xte, epochs=60, lr=1e-3, seed=0, batch=64):
    """Train on (n, ch, t) arrays (each trial z-scored per channel), return test scores."""
    import torch

    torch.manual_seed(seed); np.random.seed(seed)
    def z(X):
        return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-12)
    Xtr_t = torch.tensor(z(Xtr)[:, None], dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(z(Xte)[:, None], dtype=torch.float32)
    net = make_eegnet(Xtr.shape[1], Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(); loss = lossf(net(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xte_t).numpy()


# --------------------------------------------------------------------------- #
# 9. Reliability
# --------------------------------------------------------------------------- #
def split_half_reliability(per_trial: np.ndarray, n_split=200, seed=0):
    """per_trial: (n_subjects, n_trials) values (e.g. ERD dB per trial).
    Correlate subject means from random halves; Spearman-Brown corrects to full length:
    r_full = 2r / (1 + r). Tells whether the ERD is a stable trait-like measure."""
    rng = np.random.default_rng(seed)
    rs = []
    n = per_trial.shape[1]
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = np.nanmean(per_trial[:, p[: n // 2]], 1), np.nanmean(per_trial[:, p[n // 2:]], 1)
        r = stats.pearsonr(a, b)[0]
        rs.append(2 * r / (1 + r))
    return float(np.mean(rs))
