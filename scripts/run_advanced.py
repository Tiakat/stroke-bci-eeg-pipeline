#!/usr/bin/env python
"""Advanced analyses -> results/adv_*.csv|npz  (used by notebooks/04_advanced_eeg_analyses.ipynb)

    python scripts/run_advanced.py            # ~30-40 min on 2 CPUs

Stage 1 (per subject, parallel): spectral parameterisation, stroke markers (pdBSI, DAR),
  wPLI connectivity, time-frequency maps, per-trial ERD, covariance matrices, MRCP averages.
Stage 2 (group): cluster permutation test, cross-subject transfer, learning curves,
  EEGNet vs Riemannian, split-half reliability.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import mne  # noqa: E402

from kandu_eeg import advanced as A  # noqa: E402
from kandu_eeg import benchmark as B  # noqa: E402
from kandu_eeg import config as C  # noqa: E402
from kandu_eeg import datasets as DS  # noqa: E402
from kandu_eeg import decoding as D  # noqa: E402
from kandu_eeg import features as F  # noqa: E402
from kandu_eeg import io  # noqa: E402
from kandu_eeg import preprocessing as P  # noqa: E402
from kandu_eeg import spatial as S  # noqa: E402

mne.set_log_level("ERROR")
R = C.RESULTS_DIR
TFR_FREQS = np.arange(5, 36, 1.0)


def covs_of(X, sf):
    from pyriemann.estimation import Covariances
    return Covariances("oas").fit_transform(D.bandpass(S.car(X), sf))


# --------------------------------------------------------------------------- Liu
def liu_subject(s, part):
    warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
    hemi = part.loc[s, "contralesional_hemi"]
    aff = part.loc[s, "affected_hand"] + "_hand"
    raw, qc = P.preprocess(s)
    if not qc.attrs["usable"]:
        return None
    sf = raw.info["sfreq"]
    eeg = [c for c in raw.ch_names if c not in C.EOG_CHANNELS]
    cl, il = S.resolve("CL:C", hemi), S.resolve("IL:C", hemi)
    out = dict(subject=s, hemi=hemi)

    # ---- spectra for markers: separate 0.5-40 Hz version (delta needs < 1 Hz high-pass)
    r05 = P.filter_offline(P.remove_dc(io.load_raw(s)), l_freq=0.5, h_freq=40.0)
    r05.info["bads"] = list(qc.index[qc["bad"]])
    if r05.info["bads"]:
        r05.interpolate_bads(reset_bads=True, verbose="ERROR")
    Xc = r05.get_data(picks="eeg")
    f, pxx = signal.welch(Xc, fs=r05.info["sfreq"], nperseg=int(2 * r05.info["sfreq"]))
    names05 = r05.copy().pick("eeg").ch_names
    left = [l for l, r in C.HOMOLOGOUS_PAIRS if l in names05 and r in names05]
    right = [r for l, r in C.HOMOLOGOUS_PAIRS if l in names05 and r in names05]
    li, ri = [names05.index(c) for c in left], [names05.index(c) for c in right]
    out["pdbsi"] = A.pdbsi(f, pxx[li], pxx[ri])
    dar = A.delta_alpha_ratio(f, pxx)
    cl_side = li if hemi == "left" else ri
    il_side = ri if hemi == "left" else li
    out["dar_CL"], out["dar_IL"] = float(np.mean(dar[cl_side])), float(np.mean(dar[il_side]))
    for tag, ch in (("CL", cl), ("IL", il)):
        p = A.fooof_params(f, pxx[names05.index(ch)])
        for k, v in p.items():
            out[f"{k}_{tag}"] = v
    if s == 2:
        np.savez(R / "adv_example_spectrum.npz", f=f, psd=pxx[names05.index(cl)])

    # ---- epochs for task analyses
    ep = P.make_epochs(raw, reject_ptp=None)
    roi = [S.resolve(x, hemi) for x in ["CL:FC", "CL:C", "CL:CP", "IL:FC", "IL:C", "IL:CP"]] + ["Cz"]
    keep = P.reject_trials_ptp(ep, roi)
    sel = keep & (ep.events[:, 2] == ep.event_id[aff])
    E = S.car(ep.get_data(picks=eeg))
    times = ep.times
    tfr = A.tfr_logratio(E[sel][:, [eeg.index(cl), eeg.index(il)]], sf, times, TFR_FREQS, C.ERD_REF_WIN)
    tmask = (times >= -1.5) & (times <= 4.6)
    out["tfr"] = tfr[..., tmask][..., ::5]          # 50 Hz time grid
    out["tfr_times"] = times[tmask][::5]
    out["erd_trials"] = F.erd_per_trial(E[sel][:, [eeg.index(cl)]], sf, times, C.BANDS["mu_beta"], C.MI_WIN, C.ERD_REF_WIN)[:, 0]

    # ---- intent windows (affected hand): connectivity + covariances + raw windows
    ds = D.build_intent_dataset(ep, s, hemi, aff, "affected", keep)
    Xw = S.car(ds.X)
    for band, (lo, hi) in (("mu", (8, 13)), ("beta", (13, 30))):
        Wm = A.wpli_matrix(Xw[ds.y == 1], sf, lo, hi)
        Wr = A.wpli_matrix(Xw[ds.y == 0], sf, lo, hi)
        d = (Wm - Wr)[ds.ch_names.index(cl)]
        out[f"wpli_{band}_CLrow_mirrored"] = S.mirror_topography(dict(zip(ds.ch_names, d)), hemi)
        out[f"wpli_{band}_interhemi_move"] = Wm[ds.ch_names.index("C3"), ds.ch_names.index("C4")]
        out[f"wpli_{band}_interhemi_rest"] = Wr[ds.ch_names.index("C3"), ds.ch_names.index("C4")]
    out["y"], out["groups"] = ds.y, ds.groups
    # canonical channel order for pooling across subjects (mirrored so CL = left)
    order = [c for c in ds.ch_names]
    if hemi == "right":
        mir = {r: l for l, r in C.HOMOLOGOUS_PAIRS} | {l: r for l, r in C.HOMOLOGOUS_PAIRS}
        order = [mir.get(c, c) for c in ds.ch_names]
        idx = [order.index(c) for c in ds.ch_names]     # position of each canonical name
        out["X"] = ds.X[:, idx].astype(np.float32)
    else:
        out["X"] = ds.X.astype(np.float32)
    out["ch_names"] = ds.ch_names
    out["covs"] = covs_of(out["X"].astype(np.float64), sf)   # mirrored order -> poolable across patients
    return out


# --------------------------------------------------------------------------- MRCP + covs for other datasets
def other_subject(key, s):
    warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
    rec = DS.LOADERS[key][0](s)
    out = dict(dataset=key, subject=rec.subject)
    # MRCP grand-average data (0.3-3 Hz), time-locked to movement onset
    r = P.remove_dc(rec.raw).filter(0.3, 3.0, method="iir", iir_params=dict(order=4, ftype="butter"),
                                    phase="zero", picks=["eeg", "eog"], verbose="ERROR")
    if len(mne.pick_types(r.info, eog=True)):
        r, _ = P.eog_regression(r)
    chs = [c for c in ["Cz", "C3", "C4", "FCz"] if c in r.ch_names]
    ep = mne.Epochs(r, rec.events, tmin=-2.0, tmax=3.0, baseline=(-2.0, -1.5), picks=chs, preload=True,
                    reject_by_annotation=True, verbose="ERROR", event_repeated="drop")
    ep = ep.resample(50, verbose="ERROR") if ep.info["sfreq"] > 50 else ep
    out["mrcp"] = {c: ep.get_data(picks=[c]).mean(0)[0] for c in chs}
    out["mrcp_times"] = ep.times
    # covariances for transfer learning (one hand / pooled hands vs rest)
    raw, _ = B.preprocess(rec)
    sf = raw.info["sfreq"]
    lab = list(rec.event_id)[0] if key == "bnci2019" else "right_hand"
    X, y, g, names = B.windows(rec, raw, lab, picks=["eeg"], roi=None)
    out["covs"], out["y"], out["groups"] = covs_of(X, sf), y, g
    if key == "bnci2019":
        out["X"] = X.astype(np.float32)
    return out


CACHE = R / "cache_adv"


def cached(tag, fn, *args):
    """Run fn(*args) once and store the result on disk, so an interrupted run resumes."""
    import pickle
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{tag}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    out = fn(*args)
    f.write_bytes(pickle.dumps(out))
    return out


def main():
    t0 = time.time()
    part = io.load_participants()
    liu = [r for r in Parallel(n_jobs=2)(delayed(cached)(f"liu_{s:02d}", liu_subject, s, part) for s in range(1, 51)) if r]
    print(f"Liu stage-1 done: {len(liu)} subjects, {time.time() - t0:.0f}s", flush=True)
    # one worker: BNCI 2019 recordings are large (64 ch x ~50 min)
    oth = [cached(f"{k}_{s:02d}", other_subject, k, s) for k in ("bnci2019", "physionet_exec") for s in DS.LOADERS[k][1]]
    print(f"others stage-1 done: {len(oth)}, {time.time() - t0:.0f}s", flush=True)

    # ---------------- tables
    scal = [k for k in liu[0] if isinstance(liu[0][k], (float, int, np.floating)) and k not in ("subject",)]
    df = pd.DataFrame([{"subject": r["subject"], **{k: r[k] for k in scal}} for r in liu])
    df.to_csv(R / "adv_liu_markers.csv", index=False)
    for band in ("mu", "beta"):
        pd.DataFrame([{"subject": r["subject"], **r[f"wpli_{band}_CLrow_mirrored"]} for r in liu]).to_csv(R / f"adv_wpli_{band}_mirrored.csv", index=False)

    # ---------------- cluster test on TFR (affected-hand imagery), CL and IL
    T = np.array([r["tfr"] for r in liu])            # (n, 2, f, t)
    times = liu[0]["tfr_times"]
    res = {}
    for k, tag in enumerate(("CL", "IL")):
        tobs, sig, p = A.cluster_test_1samp(T[:, k], n_permutations=1000)
        res[f"t_{tag}"], res[f"sig_{tag}"], res[f"mean_{tag}"] = tobs, sig, T[:, k].mean(0)
    np.savez(R / "adv_tfr_cluster.npz", times=times, freqs=TFR_FREQS, **res)
    print(f"cluster test done {time.time() - t0:.0f}s", flush=True)

    # ---------------- reliability
    n = min(len(r["erd_trials"]) for r in liu)
    rel = A.split_half_reliability(np.array([r["erd_trials"][:n] for r in liu]))

    # ---------------- MRCP grand averages
    mr = {}
    for key in ("bnci2019", "physionet_exec"):
        rs = [o for o in oth if o["dataset"] == key]
        for c in ("Cz", "C3", "C4"):
            mr[f"{key}_{c}"] = np.array([o["mrcp"][c] for o in rs if c in o["mrcp"]]) * 1e6
        mr[f"{key}_times"] = rs[0]["mrcp_times"]
    np.savez(R / "adv_mrcp.npz", **mr)

    # ---------------- transfer learning (LOSO) vs within-subject
    rows = []
    for name, rs, sub in (("liu2024", liu, "subject"), ("bnci2019", [o for o in oth if o["dataset"] == "bnci2019"], "subject"),
                          ("physionet_exec", [o for o in oth if o["dataset"] == "physionet_exec"], "subject")):
        cov = {r[sub]: r["covs"] for r in rs}; yy = {r[sub]: r["y"] for r in rs}
        a_rc = A.loso_tangent(cov, yy, recentre_flag=True)
        a_raw = A.loso_tangent(cov, yy, recentre_flag=False)
        for r in rs:
            within = D.cv_auc(_ts_from_cov(r["covs"], r["y"]), None, r["y"], r["groups"])
            rows.append(dict(dataset=name, subject=r[sub], loso_recentred=a_rc[r[sub]], loso_raw=a_raw[r[sub]], within=within))
    pd.DataFrame(rows).to_csv(R / "adv_transfer.csv", index=False)
    print(f"transfer done {time.time() - t0:.0f}s", flush=True)

    # ---------------- learning curves
    lc = []
    for name, rs, sizes in (("liu2024", liu, [4, 6, 8, 10, 12, 14]),
                            ("bnci2019", [o for o in oth if o["dataset"] == "bnci2019"], [5, 10, 20, 40, 60, 80, 100])):
        for r in rs:
            for k, v in A.learning_curve(r["covs"], r["y"], r["groups"], sizes).items():
                lc.append(dict(dataset=name, subject=r["subject"], n_per_class=k, auc=v))
    pd.DataFrame(lc).to_csv(R / "adv_learning_curve.csv", index=False)
    print(f"learning curves done {time.time() - t0:.0f}s", flush=True)

    # ---------------- EEGNet vs tangent space
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    eg = []
    # (a) Liu cross-subject, 5 folds over SUBJECTS, channels mirrored to a common order
    Xs = [r["X"] for r in liu]; ys = [r["y"] for r in liu]
    subj = np.concatenate([np.full(len(y), i) for i, y in enumerate(ys)])
    Xall, yall = np.concatenate(Xs), np.concatenate(ys)
    covall = [A.recentre(r["covs"]) for r in liu]
    for k, (tr, te) in enumerate(GroupKFold(5).split(Xall, yall, subj)):
        sc = A.train_eegnet(Xall[tr], yall[tr], Xall[te], epochs=40, seed=k)
        for s_ in np.unique(subj[te]):
            m = subj[te] == s_
            eg.append(dict(dataset="liu2024", scheme="cross-subject", model="EEGNet", subject=int(s_), auc=roc_auc_score(yall[te][m], sc[m])))
    print(f"EEGNet Liu done {time.time() - t0:.0f}s", flush=True)
    # (b) BNCI2019 within-subject 5-fold (more trials per subject)
    for o in [o for o in oth if o["dataset"] == "bnci2019"]:
        X, y, g = o["X"], o["y"], o["groups"]
        sc_all = np.zeros(len(y))
        from sklearn.model_selection import StratifiedKFold
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
            sc_all[te] = A.train_eegnet(X[tr], y[tr], X[te], epochs=60, seed=k)
        eg.append(dict(dataset="bnci2019", scheme="within-subject", model="EEGNet", subject=o["subject"], auc=roc_auc_score(y, sc_all)))
        eg.append(dict(dataset="bnci2019", scheme="within-subject", model="TangentSpace", subject=o["subject"],
                       auc=D.cv_auc(_ts_from_cov(o["covs"], y), None, y, np.arange(len(y)))))
    pd.DataFrame(eg).to_csv(R / "adv_eegnet.csv", index=False)
    pd.DataFrame([dict(measure="split_half_reliability_ERD_CL", value=rel)]).to_csv(R / "adv_reliability.csv", index=False)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


def _ts_from_cov(covs, y):
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression

    def fp(tr, te):
        ts = TangentSpace(metric="riemann").fit(covs[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(ts.transform(covs[tr]), y[tr])
        return clf.decision_function(ts.transform(covs[te]))
    return fp


if __name__ == "__main__":
    main()
