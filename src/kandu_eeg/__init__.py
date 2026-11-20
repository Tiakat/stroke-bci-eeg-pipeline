"""kandu_eeg - EEG pipeline for contralesional motor-intent decoding after stroke.

Modules
-------
config         every constant, with its justification
io             raw .mat + clinical table -> MNE Raw (segment-aware)
preprocessing  DC removal, filtering, QC, resampling, EOG regression, epoching
spatial        hemisphere mapping (contralesional/ipsilesional), CAR, Laplacian, bipolar
features       PSD (Welch, AR/Burg), band power, r², ERD/ERS
decoding       intent-vs-rest datasets, decoders, leakage-safe CV, channel selection
online         causal pseudo-online controller (IpsiHand-style)
viz            plotting helpers
"""
__version__ = "1.0.0"
