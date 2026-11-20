"""Project-wide constants.

Every number used by the pipeline lives here so it can be justified in one place
(and changed in one place). Comments explain *why* each value was chosen.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw_src" / "sourcedata"          # sub-XX/sub-XX_task-motor-imagery_eeg.mat
PARTICIPANTS_TSV = DATA_ROOT / "meta" / "participants.tsv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"

# --------------------------------------------------------------------------- #
# Acquisition (from task-motor-imagery_eeg.json of the dataset)
# --------------------------------------------------------------------------- #
SFREQ_RAW = 500.0          # Hz. Nyquist = 250 Hz.
LINE_FREQ = 50.0           # Hz. Recorded in Beijing (50 Hz mains). Kandu in North America -> 60 Hz.
N_TRIALS = 40              # 20 left-hand MI + 20 right-hand MI
TRIAL_SEC = 8.0            # 2 s instruction | 4 s motor imagery (MI) | 2 s break

# Channel order inside the .mat files (33 rows). Row 18 (index 17) is CPz, the
# recording reference: it is a constant, i.e. it carries no information.
MAT_CHANNELS = [
    "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8",
    "Cz", "C3", "C4", "T3", "T4", "CPz", "CP3", "CP4", "TP7", "TP8", "Pz", "P3",
    "P4", "T5", "T6", "Oz", "O1", "O2", "HEOL", "VEOR", "TRIG",
]
REF_CHANNEL = "CPz"
EOG_CHANNELS = ["HEOL", "VEOR"]
TRIGGER_CODES = {"instruction": 1, "mi_onset": 2, "break": 3}
LABEL_CODES = {1: "left_hand", 2: "right_hand"}

# Old 10-20 names used by the dataset -> modern 10-10 names (needed for the montage)
RENAME_1010 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "FP1": "Fp1", "FP2": "Fp2"}

# --------------------------------------------------------------------------- #
# Pre-processing parameters
# --------------------------------------------------------------------------- #
HIGHPASS = 1.0      # Hz. Removes electrode drift / sweat (< 1 Hz). 1 Hz (not 0.1) is fine
                    # because we analyse oscillations, not slow ERPs/MRCPs.
LOWPASS = 40.0      # Hz. Mu (8-13) and beta (13-30) live well below 40 Hz; above 40 Hz
                    # is mostly muscle (EMG) and mains.
NOTCH_FREQS = (50.0, 100.0, 150.0, 200.0)   # mains + harmonics (only matter if LOWPASS > 50)
SFREQ_TARGET = 250.0  # Hz after downsampling. New Nyquist = 125 Hz >> 40 Hz low-pass.

# Physiologically motivated frequency bands (Hz)
BANDS = {
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),       # sensorimotor "alpha"; desynchronises (ERD) with movement / MI
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "beta": (13.0, 30.0),
    "mu_beta": (8.0, 30.0),  # the classical MI-BCI band
}

# --------------------------------------------------------------------------- #
# Epoch windows, relative to the MI-onset trigger (t = 0)
# --------------------------------------------------------------------------- #
EPOCH_TMIN, EPOCH_TMAX = -1.9, 5.0     # instruction (-2..0) | MI (0..4) | start of break; stays inside one 8-s block
MI_WIN = (0.5, 2.0)        # grand-average ERD is strongest 0.5-2 s after MI onset; skip the
                           # first 0.5 s (visual evoked response to the video onset)
REST_WIN = (-1.5, 0.0)     # instruction period (same 1.5-s length as MI_WIN). Caveat: the
                           # cue is on screen, so this is "attentive rest", not true rest.
ERD_REF_WIN = (-1.5, -0.5) # Pfurtscheller reference interval for ERD% (avoids the last
                           # 0.5 s before MI where anticipatory ERD already starts)
MIN_APERIODIC_SLOPE = -0.3 # recording-level QC: real EEG has a 1/f slope of -1..-2;
                           # ~0 means white noise (dead amplifier / lost reference)

# Peak-to-peak rejection threshold for epochs (after filtering, in volts)
REJECT_PTP_EEG = 150e-6   # applied only to the sensorimotor channels used for decoding

# --------------------------------------------------------------------------- #
# Spatial layout: hemisphere mirroring
# --------------------------------------------------------------------------- #
# (left-hemisphere, right-hemisphere) homologous pairs, 10-10 names
HOMOLOGOUS_PAIRS = [
    ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("FT7", "FT8"), ("FC3", "FC4"),
    ("T7", "T8"), ("C3", "C4"), ("TP7", "TP8"), ("CP3", "CP4"), ("P7", "P8"),
    ("P3", "P4"), ("O1", "O2"),
]
MIDLINE = ["Fz", "FCz", "Cz", "Pz", "Oz"]   # CPz is the reference -> not available

# Generic (hemisphere-agnostic) names: "F" = F3 on the left, F4 on the right, etc.
GENERIC_SITES = ["Fp", "F7/8", "F", "FT", "FC", "T", "C", "TP", "CP", "P7/8", "P", "O"]

# Montages expressed in *contralesional (CL) / ipsilesional (IL)* generic terms.
# "CL:C" means C3 if the contralesional hemisphere is the left one, else C4.
MONTAGES = {
    # Assumed Kandu-like headset: 6 electrodes, frontal-central-parietal, over the
    # UNAFFECTED hemisphere. ONE control channel (CL:C), the others feed the
    # common-average reference that removes common-mode noise.
    "kandu6": ["CL:F", "CL:FC", "CL:C", "CL:CP", "CL:P", "Cz"],
    # Montage published for the IpsiHand prototype (Bundy et al., Stroke 2017):
    # F3, F4, T7, C3, Cz, C4, T8, Pz with common-average reference.
    "bundy8": ["F3", "F4", "T7", "C3", "Cz", "C4", "T8", "Pz"],
}
CONTROL_SITE = "CL:C"

RANDOM_STATE = 42
