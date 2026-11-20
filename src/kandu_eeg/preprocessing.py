"""Pre-processing: DC removal, filtering, quality control, resampling, epoching.

Order matters:

    1. DC offset removal      - the amplifier is DC-coupled; offsets are ~10 mV.
    2. High-pass  (1 Hz)      - remove drift BEFORE anything that assumes zero-mean.
    3. Notch      (50 Hz...)  - mains. Optional if the low-pass is well below 50 Hz.
    4. Low-pass   (40 Hz)     - also acts as the anti-aliasing filter for step 6.
    5. Bad-channel detection  - on filtered data (drift would make every channel "bad").
    6. Downsample (500->250)  - only AFTER the low-pass, otherwise aliasing.
    7. Artefact handling      - EOG regression; epoch rejection by peak-to-peak.
    8. Re-reference           - CAR / Laplacian / bipolar (see spatial.py).
    9. Epoch                  - cut around MI onset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
from scipy import signal

from . import config as C


# --------------------------------------------------------------------------- #
# 1-4. Filtering
# --------------------------------------------------------------------------- #
def remove_dc(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Subtract each channel's mean, segment by segment.

    The amplifier is DC-coupled: offsets are ~2-15 mV, i.e. 100-1000x the EEG.
    A filter started on a 10 mV offset produces a huge onset transient.
    """
    from .io import segments

    raw = raw.copy()
    for a, b in segments(raw):
        raw._data[:, a:b] -= raw._data[:, a:b].mean(axis=1, keepdims=True)
    return raw


def filter_offline(
    raw: mne.io.BaseRaw,
    l_freq: float = C.HIGHPASS,
    h_freq: float = C.LOWPASS,
    notch: tuple | None = C.NOTCH_FREQS,
) -> mne.io.BaseRaw:
    """Offline (zero-phase) filtering: FIR, Hamming window, `firwin` design.

    Zero-phase = the filter is applied forward and its delay compensated, so ERD
    timing is not shifted. Only possible offline (it needs future samples).
    """
    raw = raw.copy()
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    if notch:
        freqs = [f for f in notch if f < raw.info["sfreq"] / 2]
        raw.notch_filter(freqs, picks=picks, method="fir", verbose="ERROR")
    raw.filter(
        l_freq, h_freq, picks=picks, method="fir", phase="zero",
        fir_window="hamming", fir_design="firwin", verbose="ERROR",
    )
    return raw


@dataclass
class CausalBandpass:
    """Online-style (causal) IIR band-pass with persistent state.

    This is what runs on a device: each new chunk of samples is filtered using only
    past samples (`sosfilt` with `zi`). Price: phase delay (see group_delay()).
    """
    sfreq: float
    l_freq: float = 8.0
    h_freq: float = 30.0
    order: int = 4
    sos: np.ndarray = field(init=False)
    zi: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        self.sos = signal.butter(self.order, [self.l_freq, self.h_freq], btype="bandpass",
                                 fs=self.sfreq, output="sos")

    def reset(self, n_channels: int):
        zi = signal.sosfilt_zi(self.sos)                    # (n_sections, 2)
        self.zi = np.repeat(zi[:, None, :], n_channels, axis=1)  # (n_sections, n_ch, 2)
        self.zi *= 0.0

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (n_channels, n_samples) -> filtered chunk, state carried over."""
        if self.zi is None:
            self.reset(chunk.shape[0])
        out, self.zi = signal.sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        return out

    def group_delay(self, freqs: np.ndarray) -> np.ndarray:
        """Group delay in seconds at `freqs` (Hz)."""
        b, a = signal.sos2tf(self.sos)
        w, gd = signal.group_delay((b, a), w=freqs, fs=self.sfreq)
        return gd / self.sfreq


# --------------------------------------------------------------------------- #
# 5. Channel quality control
# --------------------------------------------------------------------------- #
def channel_qc(raw: mne.io.BaseRaw, z_thresh: float = 4.0, corr_thresh: float = 0.2) -> "pd.DataFrame":
    """Per-channel quality metrics on *filtered* data. Returns a DataFrame.

    - std_uv: overall amplitude. Robust z-score (median/MAD) flags outliers.
    - line_ratio: power at mains / power in 1-40 Hz on the *unfiltered* data tells you
      about electrode impedance (high impedance -> more mains pickup).
    - corr_neighbours: median correlation with the 3 nearest electrodes. A good
      channel correlates with its neighbours (volume conduction); a bridged channel
      correlates ~1, a dead/loose one ~0. NB: channels next to the reference
      (Cz, Pz next to CPz) are naturally small and less correlated -> lenient threshold.
    - flat: std below 0.5 µV.
    """
    import pandas as pd

    eeg = raw.copy().pick("eeg")
    X = eeg.get_data()
    std = X.std(axis=1)
    med, mad = np.median(std), np.median(np.abs(std - np.median(std))) * 1.4826
    z = (std - med) / (mad + 1e-20)

    pos = np.array([eeg.info["chs"][i]["loc"][:3] for i in range(len(eeg.ch_names))])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    R = np.corrcoef(X)
    corr_nb = []
    for i in range(len(pos)):
        nb = np.argsort(d[i])[1:4]
        corr_nb.append(np.median(R[i, nb]))
    df = pd.DataFrame(
        {"std_uv": std * 1e6, "robust_z": z, "corr_neighbours": corr_nb},
        index=eeg.ch_names,
    )
    df["flat"] = df["std_uv"] < 0.5
    df["noisy"] = df["robust_z"] > z_thresh
    df["uncorrelated"] = df["corr_neighbours"] < corr_thresh
    df["bad"] = df[["flat", "noisy", "uncorrelated"]].any(axis=1)
    return df


def line_noise_ratio(raw_unfiltered: mne.io.BaseRaw, f0: float = C.LINE_FREQ) -> "pd.Series":
    """10*log10(PSD at f0 / median PSD 1-40 Hz) per channel (dB). High -> bad contact."""
    import pandas as pd

    eeg = raw_unfiltered.copy().pick("eeg")
    f, p = signal.welch(eeg.get_data(), fs=eeg.info["sfreq"], nperseg=int(2 * eeg.info["sfreq"]))
    i0 = np.argmin(np.abs(f - f0))
    band = (f >= 1) & (f <= 40)
    return pd.Series(10 * np.log10(p[:, i0] / np.median(p[:, band], axis=1)), index=eeg.ch_names)


def aperiodic_slope(raw: mne.io.BaseRaw, fmin: float = 2.0, fmax: float = 40.0) -> float:
    """Median (over channels) slope of log10 PSD vs log10 f between fmin and fmax.

    Real EEG has a 1/f^chi spectrum (slope ~ -1 to -2). A slope near 0 means the
    "EEG" is white noise: amplifier saturation, lost reference/ground, etc.
    Recording-level check: if it fails, the whole session is unusable.
    """
    eeg = raw.copy().pick("eeg")
    f, p = signal.welch(eeg.get_data(), fs=eeg.info["sfreq"], nperseg=int(2 * eeg.info["sfreq"]))
    m = (f >= fmin) & (f <= fmax) & (np.abs(f - C.LINE_FREQ) > 2)
    slopes = [np.polyfit(np.log10(f[m]), np.log10(pp[m]), 1)[0] for pp in p]
    return float(np.median(slopes))


# --------------------------------------------------------------------------- #
# 6. Downsampling
# --------------------------------------------------------------------------- #
def downsample(raw: mne.io.BaseRaw, sfreq: float = C.SFREQ_TARGET) -> mne.io.BaseRaw:
    """Polyphase resampling (anti-alias low-pass built in), segment by segment.

    500 -> 250 Hz is an integer factor (2): polyphase = low-pass FIR + keep every
    2nd sample. Done per EDGE segment so the anti-alias filter never crosses a join.
    """
    from .io import segments

    if raw.info["sfreq"] == sfreq:
        return raw.copy()
    pieces = []
    for a, b in segments(raw):
        seg = raw.copy().crop(a / raw.info["sfreq"], (b - 1) / raw.info["sfreq"], include_tmax=True)
        pieces.append(seg.resample(sfreq, method="polyphase", verbose="ERROR"))
    out = mne.concatenate_raws(pieces, verbose="ERROR")
    # concatenate_raws adds 'BAD boundary' (zero-length) markers; keep only the EDGE ones
    a = out.annotations
    seen, keep = set(), []
    for i, (o, d) in enumerate(zip(a.onset, a.description)):
        if d == "BAD boundary":
            continue
        key = (round(o, 3), d)
        if d.startswith("EDGE") and key in seen:
            continue
        seen.add(key)
        keep.append(i)
    out.set_annotations(a[keep])
    return out


# --------------------------------------------------------------------------- #
# 7. Artefacts
# --------------------------------------------------------------------------- #
def eog_regression(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, np.ndarray]:
    """Remove the part of each EEG channel that is linearly explained by the EOG
    channels (Gratton/Croft-style least-squares regression).

        EEG_clean = EEG - B @ EOG,   B = EEG EOG^T (EOG EOG^T)^-1

    Why regression and not ICA? With 6 electrodes (Kandu-like) ICA is not viable
    (it needs many channels to separate sources). Regression only needs EOG
    references, is deterministic, and cheap enough to run online.
    Caveat: EOG channels also pick up some frontal EEG, so regression removes a
    little brain signal too (mostly frontal, negligible at C3/C4).
    Returns (cleaned raw, B [n_eeg, n_eog]).
    """
    raw = raw.copy()
    eeg = mne.pick_types(raw.info, eeg=True, exclude="bads")
    eog = mne.pick_types(raw.info, eog=True)
    Y, E = raw._data[eeg], raw._data[eog]
    B = Y @ E.T @ np.linalg.inv(E @ E.T)
    raw._data[eeg] = Y - B @ E
    return raw, B


# --------------------------------------------------------------------------- #
# 9. Epoching
# --------------------------------------------------------------------------- #
def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = C.EPOCH_TMIN,
    tmax: float = C.EPOCH_TMAX,
    reject_ptp: float | None = C.REJECT_PTP_EEG,
    baseline=None,
) -> mne.Epochs:
    from .io import mi_events

    events, ev_id = mi_events(raw)
    reject = dict(eeg=reject_ptp) if reject_ptp else None
    return mne.Epochs(
        raw, events, ev_id, tmin=tmin, tmax=tmax, baseline=baseline, reject=reject,
        preload=True, picks=["eeg", "eog"], verbose="ERROR",
    )


def reject_trials_ptp(epochs: mne.Epochs, picks: list[str], thresh: float = C.REJECT_PTP_EEG) -> np.ndarray:
    """Boolean mask of trials to KEEP: peak-to-peak below `thresh` on `picks` only.

    We only judge the channels the decoder uses: a blink on Fp1 should not throw
    away a trial whose C3/C4 are clean (frontal/temporal channels carry EOG/EMG).
    """
    X = epochs.get_data(picks=picks)
    ptp = X.max(-1) - X.min(-1)
    return (ptp < thresh).all(axis=1)


def preprocess(subject: int, sfreq: float = C.SFREQ_TARGET, eog: bool = True,
               interpolate_bads: bool = True) -> tuple[mne.io.BaseRaw, "pd.DataFrame"]:
    """The full offline chain for one patient. Returns (clean raw, channel QC table).

    qc.attrs holds recording-level QC: 'aperiodic_slope' and 'usable'.
    """
    from .io import load_raw

    raw = load_raw(subject)
    raw = remove_dc(raw)                                   # 1
    raw = filter_offline(raw)                              # 2-4 (per segment)
    slope = aperiodic_slope(raw)                           # recording-level QC
    qc = channel_qc(raw)                                   # 5
    qc.attrs.update(aperiodic_slope=slope, usable=slope < C.MIN_APERIODIC_SLOPE)
    raw.info["bads"] = list(qc.index[qc["bad"]])
    raw = downsample(raw, sfreq)                           # 6
    if eog:
        raw, _ = eog_regression(raw)                       # 7
    if interpolate_bads and raw.info["bads"] and qc.attrs["usable"]:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")   # spherical splines
    return raw, qc
