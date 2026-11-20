#!/usr/bin/env python
"""Q8 - Is the MI-vs-rest decoder reading MOTOR cortex, or eyes / vision?

In this dataset "MI" = watching a gripping video while imagining, "rest" = looking at a
static cue picture. So MI-vs-rest can also be decoded from visual (occipital alpha
blocking) or ocular (eye-movement) differences. We decode with channel groups that
cannot see the motor cortex well and compare:

  sensorimotor : FC3 C3 CP3 FC4 C4 CP4      (motor strip, both hemispheres)
  occipital    : O1 Oz O2 P7 P8              (visual cortex)
  frontal-pole : Fp1 Fp2 F7 F8               (eyes / forehead muscle)
  eog          : HEOL VEOR                   (pure eye channels; no brain)
Features: log mu / low-beta / high-beta power, CAR-29, shrinkage LDA, same CV.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import mne  # noqa: E402

from kandu_eeg import config as C, decoding as D, features as F, io, preprocessing as P, spatial as S  # noqa: E402

mne.set_log_level("ERROR")
GROUPS = {"sensorimotor": ["FC3", "C3", "CP3", "FC4", "C4", "CP4"], "occipital": ["O1", "Oz", "O2", "P7", "P8"],
          "frontal_pole": ["Fp1", "Fp2", "F7", "F8"]}
BANDS = {b: C.BANDS[b] for b in ["mu", "low_beta", "high_beta"]}


def one(s, part):
    hemi = part.loc[s, "contralesional_hemi"]; aff = part.loc[s, "affected_hand"] + "_hand"
    raw, qc = P.preprocess(s)
    if not qc.attrs["usable"]:
        return []
    ep = P.make_epochs(raw, reject_ptp=None)
    roi = [S.resolve(x, hemi) for x in ["CL:FC", "CL:C", "CL:CP", "IL:FC", "IL:C", "IL:CP"]] + ["Cz"]
    keep = P.reject_trials_ptp(ep, roi)
    ds = D.build_intent_dataset(ep, s, hemi, aff, "affected", keep)
    Xc = S.car(ds.X)
    rows = []
    for g, chs in GROUPS.items():
        f = D.bandpower_feats(Xc[:, ds.idx(chs)], ds.sfreq)
        rows.append(dict(subject=s, group=g, auc=D.cv_auc(D.lda_decoder(f, ds.y), None, ds.y, ds.groups)))
    # EOG channels (after regression they are the untouched EOG references)
    sel = ep.events[:, 2] == ep.event_id[aff]
    sel &= keep
    E = ep.get_data(picks="eog")[sel]
    t = ep.times
    mi = (t >= C.MI_WIN[0]) & (t < C.MI_WIN[1]); rs = (t >= C.REST_WIN[0]) & (t < C.REST_WIN[1]); n = min(mi.sum(), rs.sum())
    Xe = np.concatenate([E[..., np.flatnonzero(rs)[:n]], E[..., np.flatnonzero(mi)[:n]]])
    # EOG features: log power 1-4 Hz (eye movements) + mu/beta, per channel
    fe = np.concatenate([F.band_log_power(Xe, ds.sfreq, {"slow": (1, 4), **BANDS}).reshape(len(Xe), -1)], 1)
    rows.append(dict(subject=s, group="eog", auc=D.cv_auc(D.lda_decoder(fe, ds.y), None, ds.y, ds.groups)))

    # --- Control task: AFFECTED vs UNAFFECTED hand imagery, both in the MI window.
    # Same video, same cue timing -> ocular/visual confounds cancel; only "which hand"
    # differs, which must come from (lateralised) motor cortex activity.
    eeg = [c for c in ep.ch_names if c not in C.EOG_CHANNELS]
    lab = ep.events[:, 2]
    m = keep
    Xa = S.car(ep.get_data(picks=eeg)[m][..., np.flatnonzero(mi)])
    Ea = ep.get_data(picks="eog")[m][..., np.flatnonzero(mi)]
    yh = (lab[m] == ep.event_id[aff]).astype(int)
    gh = np.arange(m.sum())
    for g, chs in GROUPS.items():
        f = D.bandpower_feats(Xa[:, [eeg.index(c) for c in chs]], ds.sfreq)
        rows.append(dict(subject=s, group=g, task="hand_vs_hand", auc=D.cv_auc(D.lda_decoder(f, yh), None, yh, gh)))
    fe = F.band_log_power(Ea, ds.sfreq, {"slow": (1, 4), **BANDS}).reshape(len(Ea), -1)
    rows.append(dict(subject=s, group="eog", task="hand_vs_hand", auc=D.cv_auc(D.lda_decoder(fe, yh), None, yh, gh)))
    for r in rows:
        r.setdefault("task", "mi_vs_rest")
    return rows


if __name__ == "__main__":
    part = io.load_participants()
    res = Parallel(n_jobs=2)(delayed(one)(s, part) for s in range(1, 51))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(C.RESULTS_DIR / "confound_check.csv", index=False)
    print(df.groupby(["task", "group"]).auc.describe().round(3))
