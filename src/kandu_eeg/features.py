"""Spectral features + the ERD/ERS computation.

ERD/ERS (Pfurtscheller & Lopes da Silva, 1999)
---------------------------------------------
ERD% (t) = (A(t) - R) / R * 100
    A(t): band power at time t, averaged across trials
    R   : mean band power in a reference (rest) interval
Negative = desynchronisation (ERD, "the cortex is busy"), positive = ERS.
Classic recipe: band-pass -> square (power) -> average over trials -> smooth -> % vs ref.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


# --------------------------------------------------------------------------- #
# Power spectra
# --------------------------------------------------------------------------- #
def welch_psd(X: np.ndarray, sfreq: float, nperseg: int | None = None, fmax: float = 40.0):
    """Welch PSD along the last axis. Returns freqs, psd (V^2/Hz).

    Welch = average of |FFT|^2 over overlapping, windowed (Hann) segments ->
    lower variance than one FFT, at the cost of frequency resolution
    (resolution = sfreq / nperseg; 250 Hz / 250 samples = 1 Hz).
    """
    nperseg = nperseg or int(sfreq)
    nperseg = min(nperseg, X.shape[-1])
    f, p = signal.welch(X, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    keep = f <= fmax
    return f[keep], p[..., keep]


def burg_ar(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Burg's method for AR coefficients (1-D). Returns a (len order+1, a[0]=1), noise var."""
    x = np.asarray(x, float)
    N = len(x)
    a = np.array([1.0])
    ef, eb = x[1:].copy(), x[:-1].copy()
    E = np.dot(x, x) / N
    for m in range(order):
        num = -2.0 * np.dot(eb, ef)
        den = np.dot(ef, ef) + np.dot(eb, eb)
        k = num / den
        a = np.r_[a, 0.0] + k * np.r_[0.0, a[::-1]]
        ef, eb = ef[1:] + k * eb[1:], eb[:-1] + k * ef[:-1]
        E *= 1 - k * k
    return a, E


def ar_psd(X: np.ndarray, sfreq: float, order: int = 16, freqs: np.ndarray | None = None):
    """Maximum-entropy (AR/Burg) PSD, as used by BCI2000 and by the IpsiHand study.

    Why AR for BCI? With short windows (e.g. 500 ms = 125 samples) an FFT has only
    2 Hz resolution and is very noisy; an AR model gives a smooth spectrum with
    sharp peaks from few samples.
    """
    freqs = np.arange(1, 41, 1.0) if freqs is None else freqs
    flat = X.reshape(-1, X.shape[-1])
    out = np.empty((flat.shape[0], len(freqs)))
    z = np.exp(-2j * np.pi * freqs / sfreq)
    for i, x in enumerate(flat):
        a, E = burg_ar(x - x.mean(), order)
        H = np.polyval(a[::-1], z)          # sum a_k z^k
        out[i] = E / np.abs(H) ** 2 / sfreq
    return freqs, out.reshape(*X.shape[:-1], len(freqs))


def band_log_power(X: np.ndarray, sfreq: float, bands: dict[str, tuple[float, float]]):
    """log10 band power per band -> (..., n_bands). log because power is right-skewed
    (log makes it ~Gaussian, which suits LDA)."""
    f, p = welch_psd(X, sfreq)
    cols = []
    for lo, hi in bands.values():
        m = (f >= lo) & (f < hi)
        cols.append(np.log10(p[..., m].mean(axis=-1)))
    return np.stack(cols, axis=-1)


# --------------------------------------------------------------------------- #
# Feature ranking
# --------------------------------------------------------------------------- #
def signed_r2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed squared point-biserial correlation between two conditions (BCI2000 'r²').
    a: (n_a, ...) features of condition A (e.g. MI), b: (n_b, ...) condition B (rest).
    Negative sign = feature lower in A (ERD). |r²| = fraction of variance explained by the label.
    """
    n1, n2 = len(a), len(b)
    allx = np.concatenate([a, b], axis=0)
    r = (a.mean(0) - b.mean(0)) * np.sqrt(n1 * n2) / ((n1 + n2) * allx.std(0) + 1e-30)
    return np.sign(r) * r ** 2


# --------------------------------------------------------------------------- #
# ERD / ERS
# --------------------------------------------------------------------------- #
def erd_timecourse(
    epochs_data: np.ndarray, sfreq: float, times: np.ndarray, band: tuple[float, float],
    ref_win: tuple[float, float], smooth_sec: float = 0.25, order: int = 4,
):
    """Classical Pfurtscheller ERD%.

    epochs_data: (n_trials, n_channels, n_times) - *broadband* filtered epochs
    returns (n_channels, n_times) ERD% (negative = ERD)
    """
    sos = signal.butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    xb = signal.sosfiltfilt(sos, epochs_data, axis=-1)        # 1. band-pass (zero-phase)
    pw = xb ** 2                                              # 2. square -> instantaneous power
    # 3. subtract the evoked (phase-locked) part -> "inter-trial variance" method
    pw = pw - (xb.mean(axis=0, keepdims=True)) ** 2
    A = pw.mean(axis=0)                                       # 4. average over trials
    w = max(1, int(smooth_sec * sfreq))
    A = uniform_filter1d(A, size=w, axis=-1, mode="nearest")  # 5. moving-average smoothing
    m = (times >= ref_win[0]) & (times < ref_win[1])
    R = A[:, m].mean(axis=1, keepdims=True)
    return (A - R) / R * 100.0                                # 6. percent change vs reference


def erd_per_trial(epochs_data, sfreq, times, band, win, ref_win):
    """Per-trial log-ratio of band power in `win` vs `ref_win` (dB). Used for statistics."""
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    xb = signal.sosfiltfilt(sos, epochs_data, axis=-1)
    pw = xb ** 2
    a = pw[..., (times >= win[0]) & (times < win[1])].mean(-1)
    r = pw[..., (times >= ref_win[0]) & (times < ref_win[1])].mean(-1)
    return 10 * np.log10(a / r)
