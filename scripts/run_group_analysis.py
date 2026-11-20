#!/usr/bin/env python
"""Group analysis over all patients -> results/*.csv|npz.

Usage
-----
    python scripts/run_group_analysis.py              # all 50 patients, 2 workers
    python scripts/run_group_analysis.py --subjects 1 2 3 --n-jobs 1

Research questions addressed (see notebooks/02_group_analysis.ipynb):
  Q1  Data quality: which recordings / channels / trials are usable?
  Q2  Is there an ERD in the CONTRALESIONAL hemisphere when patients imagine moving
      the AFFECTED hand (the physiological premise of IpsiHand)?
  Q3  How well can intent (MI vs rest) be decoded from ONE contralesional electrode,
      and how much does the reference (raw / CAR6 / CAR8 / CAR29 / Laplacian /
      bipolar) matter?  -> how many electrodes do we really need?
  Q4  Does a data-driven channel subset beat the fixed montage? AUC vs #channels.
  Q5  Which frequency carries the intent signal (patient-specific bins vs fixed band)?
  Q6  Window length (latency) vs accuracy; sampling rate vs accuracy.
  Q7  Is decodability related to clinical scores (NIHSS, MBI, mRS, time since stroke,
      cortical involvement)?
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import mne  # noqa: E402

from kandu_eeg import config as C  # noqa: E402
from kandu_eeg import decoding as D  # noqa: E402
from kandu_eeg import features as F  # noqa: E402
from kandu_eeg import io  # noqa: E402
from kandu_eeg import preprocessing as P  # noqa: E402
from kandu_eeg import spatial as S  # noqa: E402

mne.set_log_level("ERROR")

DERIVATIONS = ["raw", "bip_cp", "bip_fc", "lap4", "car6", "car8", "car29"]
N_ELECTRODES = {"raw": 2, "bip_cp": 2, "bip_fc": 2, "lap4": 5, "car6": 6, "car8": 8, "car29": 29}


def roi_channels(hemi):
    return [S.resolve(x, hemi) for x in ["CL:FC", "CL:C", "CL:CP", "IL:FC", "IL:C", "IL:CP"]] + ["Cz"]


def analyse_subject(s: int, part: pd.DataFrame, n_perm: int = 200) -> dict:
    t0 = time.time()
    hemi = part.loc[s, "contralesional_hemi"]
    aff = part.loc[s, "affected_hand"] + "_hand"
    unaff = ("right" if part.loc[s, "affected_hand"] == "left" else "left") + "_hand"

    raw, qc = P.preprocess(s)
    out = {"subject": s}
    qrow = dict(subject=s, aperiodic_slope=qc.attrs["aperiodic_slope"], usable=qc.attrs["usable"],
                n_bad_channels=int(qc["bad"].sum()), bad_channels=",".join(qc.index[qc["bad"]]))
    if not qc.attrs["usable"]:
        out["qc"] = qrow
        return out

    ep = P.make_epochs(raw, reject_ptp=None)
    keep = P.reject_trials_ptp(ep, roi_channels(hemi))
    qrow.update(n_trials_total=len(ep), n_trials_kept=int(keep.sum()),
                n_aff_kept=int((keep & (ep.events[:, 2] == ep.event_id[aff])).sum()))
    out["qc"] = qrow
    eeg = [c for c in ep.ch_names if c not in C.EOG_CHANNELS]
    times = ep.times
    sf = ep.info["sfreq"]
    Xall = ep.get_data(picks=eeg)

    # ------------------------------------------------------------------ Q2: ERD
    erd = {}
    Xcar = S.car(Xall)
    cl, il = eeg.index(S.resolve("CL:C", hemi)), eeg.index(S.resolve("IL:C", hemi))
    for hand_name, lab in [("affected", aff), ("unaffected", unaff)]:
        sel = keep & (ep.events[:, 2] == ep.event_id[lab])
        for bname in ["mu", "beta"]:
            tc = F.erd_timecourse(Xcar[sel], sf, times, C.BANDS[bname], C.ERD_REF_WIN)
            erd[f"{hand_name}_{bname}"] = tc[[cl, il]]
        # per-trial ERD (dB) in MI window, all channels, mu+beta -> topography
        e = F.erd_per_trial(Xcar[sel], sf, times, C.BANDS["mu_beta"], C.MI_WIN, C.ERD_REF_WIN)
        topo = dict(zip(eeg, np.median(e, axis=0)))
        erd[f"{hand_name}_topo_mirrored"] = S.mirror_topography(topo, hemi)
        erd[f"{hand_name}_erd_db_CL"] = float(np.median(e[:, cl]))
        erd[f"{hand_name}_erd_db_IL"] = float(np.median(e[:, il]))
    out["erd"] = erd

    # ------------------------------------------------------------------ Q3: single channel
    ds = D.build_intent_dataset(ep, s, hemi, aff, "affected", keep)
    ds_un = D.build_intent_dataset(ep, s, hemi, unaff, "unaffected", keep)
    rows = []
    for der in DERIVATIONS:
        for site in ["CL:C", "IL:C"]:
            x = D.control_signal(ds, der, site)
            f1 = D.spectrum_1hz(x, sf)
            fb = F.band_log_power(x, sf, {"mu_beta": C.BANDS["mu_beta"]})
            rows.append(dict(subject=s, hand="affected", pipeline=f"single_{der}_bestbin", site=site,
                             n_electrodes=N_ELECTRODES[der], auc=D.cv_auc(D.best_bin_decoder(f1, ds.y), None, ds.y, ds.groups)))
            rows.append(dict(subject=s, hand="affected", pipeline=f"single_{der}_mubeta", site=site,
                             n_electrodes=N_ELECTRODES[der], auc=D.cv_auc(D.lda_decoder(fb, ds.y, shrink=False), None, ds.y, ds.groups)))
    # unaffected hand: is the ipsilateral (contralesional) signal specific to the affected hand?
    for site in ["CL:C", "IL:C"]:
        x = D.control_signal(ds_un, "car29", site)
        f1 = D.spectrum_1hz(x, sf)
        rows.append(dict(subject=s, hand="unaffected", pipeline="single_car29_bestbin", site=site,
                         n_electrodes=29, auc=D.cv_auc(D.best_bin_decoder(f1, ds_un.y), None, ds_un.y, ds_un.groups)))

    # multichannel
    mont6 = S.montage_channels("kandu6", hemi)
    X6 = S.car(ds.X[:, ds.idx(mont6)])
    rows.append(dict(subject=s, hand="affected", pipeline="kandu6_bandpower_lda", site="multi", n_electrodes=6,
                     auc=D.cv_auc(D.lda_decoder(D.bandpower_feats(X6, sf), ds.y), None, ds.y, ds.groups)))
    Xb6 = D.bandpass(X6, sf)
    rows.append(dict(subject=s, hand="affected", pipeline="kandu6_tangent_space", site="multi", n_electrodes=6,
                     auc=D.cv_auc(D.tangent_space_decoder(Xb6, ds.y), None, ds.y, ds.groups)))
    Xb = D.bandpass(ds.X, sf)
    rows.append(dict(subject=s, hand="affected", pipeline="all29_csp_lda", site="multi", n_electrodes=29,
                     auc=D.cv_auc(D.csp_decoder(Xb, ds.y), None, ds.y, ds.groups)))
    rows.append(dict(subject=s, hand="affected", pipeline="all29_tangent_space", site="multi", n_electrodes=29,
                     auc=D.cv_auc(D.tangent_space_decoder(Xb, ds.y), None, ds.y, ds.groups)))
    out["auc"] = rows

    # permutation p-values for the IpsiHand-like decoder and the best multichannel one
    x = D.control_signal(ds, "car6", "CL:C")
    f1 = D.spectrum_1hz(x, sf)
    obs = [r["auc"] for r in rows if r["pipeline"] == "single_car6_bestbin" and r["site"] == "CL:C"][0]
    p_car6 = D.permutation_pvalue(lambda f, y: D.best_bin_decoder(f, y), f1, ds.y, ds.groups, obs, n_perm=n_perm, seed=s)
    out["perm"] = dict(subject=s, auc_car6=obs, p_car6=p_car6)

    # ------------------------------------------------------------------ Q5: best frequency
    r2 = F.signed_r2(f1[ds.y == 1], f1[ds.y == 0])
    out["freq"] = dict(subject=s, best_freq=float(D.FREQS[np.argmin(r2)]), best_r2=float(r2.min()),
                       r2_spectrum=r2.tolist())

    # ------------------------------------------------------------------ Q4: single-channel map + greedy
    feats_ch = F.band_log_power(S.car(ds.X), sf, {b: C.BANDS[b] for b in ["mu", "low_beta", "high_beta"]})
    single = {}
    for i, ch in enumerate(eeg):
        single[ch] = D.cv_auc(D.lda_decoder(feats_ch[:, i], ds.y), None, ds.y, ds.groups, n_repeats=3)
    out["single_map_mirrored"] = S.mirror_topography(single, hemi)
    curve, counts = D.greedy_selection_curve(feats_ch, ds.y, ds.groups, k_max=6, n_repeats=2)
    rel = S.lesion_relative_names(eeg, hemi)
    out["greedy"] = dict(subject=s, auc_k=curve.tolist(),
                         first_pick=rel.get(eeg[int(np.argmax(counts[0]))], eeg[int(np.argmax(counts[0]))]),
                         counts_rel={rel.get(eeg[i], eeg[i]): float(counts[:, i].sum()) for i in range(len(eeg))})

    # ------------------------------------------------------------------ Q6: window length
    wl = []
    t = ep.times
    Daff = ep.get_data(picks=eeg)[keep & (ep.events[:, 2] == ep.event_id[aff])]
    for L in [0.5, 1.0, 1.5]:
        mi = (t >= 0.5) & (t < 0.5 + L)
        rs = (t >= -L) & (t < 0)
        n = min(mi.sum(), rs.sum())
        Xw = np.concatenate([Daff[..., np.flatnonzero(rs)[:n]], Daff[..., np.flatnonzero(mi)[:n]]])
        yw = np.r_[np.zeros(len(Daff), int), np.ones(len(Daff), int)]
        gw = np.r_[np.arange(len(Daff)), np.arange(len(Daff))]
        dsw = D.IntentDataset(s, Xw, yw, gw, eeg, sf, hemi, "affected")
        xw = D.control_signal(dsw, "car6", "CL:C")
        for method in ["welch", "ar"]:
            fw = D.spectrum_1hz(xw, sf, method=method)
            wl.append(dict(subject=s, window_s=L, psd=method,
                           auc=D.cv_auc(D.best_bin_decoder(fw, yw), None, yw, gw)))
    out["window"] = wl

    # ------------------------------------------------------------------ Q6b: sampling rate
    sr = []
    for fs_t in [125.0, 500.0]:
        raw_b, _ = P.preprocess(s, sfreq=fs_t)
        ep_b = P.make_epochs(raw_b, reject_ptp=None)
        ds_b = D.build_intent_dataset(ep_b, s, hemi, aff, "affected", keep)
        xb = D.control_signal(ds_b, "car6", "CL:C")
        fbb = D.spectrum_1hz(xb, fs_t)
        sr.append(dict(subject=s, sfreq=fs_t, auc=D.cv_auc(D.best_bin_decoder(fbb, ds_b.y), None, ds_b.y, ds_b.groups)))
    sr.append(dict(subject=s, sfreq=250.0, auc=obs))
    out["sfreq"] = sr
    out["elapsed"] = time.time() - t0
    print(f"sub-{s:02d} done in {out['elapsed']:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 51)))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    part = io.load_participants()
    res = Parallel(n_jobs=args.n_jobs)(delayed(analyse_subject)(s, part, args.n_perm) for s in args.subjects)

    R = C.RESULTS_DIR
    R.mkdir(exist_ok=True)
    pd.DataFrame([r["qc"] for r in res]).to_csv(R / "qc_summary.csv", index=False)
    ok = [r for r in res if "auc" in r]
    pd.DataFrame([row for r in ok for row in r["auc"]]).to_csv(R / "decoding_auc.csv", index=False)
    pd.DataFrame([r["perm"] for r in ok]).to_csv(R / "permutation.csv", index=False)
    fr = pd.DataFrame([{k: v for k, v in r["freq"].items() if k != "r2_spectrum"} for r in ok])
    fr.to_csv(R / "best_frequency.csv", index=False)
    np.save(R / "r2_spectra.npy", np.array([r["freq"]["r2_spectrum"] for r in ok]))
    pd.DataFrame([row for r in ok for row in r["window"]]).to_csv(R / "window_length.csv", index=False)
    pd.DataFrame([row for r in ok for row in r["sfreq"]]).to_csv(R / "sampling_rate.csv", index=False)
    g = pd.DataFrame([{"subject": r["greedy"]["subject"], "first_pick": r["greedy"]["first_pick"],
                       **{f"k{k + 1}": a for k, a in enumerate(r["greedy"]["auc_k"])}} for r in ok])
    g.to_csv(R / "greedy_selection.csv", index=False)
    pd.DataFrame([{"subject": r["subject"], **r["greedy"]["counts_rel"]} for r in ok]).to_csv(R / "greedy_counts.csv", index=False)
    pd.DataFrame([{"subject": r["subject"], **r["single_map_mirrored"]} for r in ok]).to_csv(R / "single_channel_auc_mirrored.csv", index=False)

    # ERD
    keys = ["affected_mu", "affected_beta", "unaffected_mu", "unaffected_beta"]
    np.savez(R / "erd_timecourses.npz", subjects=np.array([r["subject"] for r in ok]),
             **{k: np.array([r["erd"][k] for r in ok]) for k in keys})
    pd.DataFrame([{"subject": r["subject"], **r["erd"]["affected_topo_mirrored"]} for r in ok]).to_csv(R / "erd_topo_affected_mirrored.csv", index=False)
    pd.DataFrame([{"subject": r["subject"], **r["erd"]["unaffected_topo_mirrored"]} for r in ok]).to_csv(R / "erd_topo_unaffected_mirrored.csv", index=False)
    pd.DataFrame([{"subject": r["subject"], **{k: v for k, v in r["erd"].items() if k.endswith(("_CL", "_IL"))}} for r in ok]).to_csv(R / "erd_db.csv", index=False)
    print("saved to", R)


if __name__ == "__main__":
    main()
