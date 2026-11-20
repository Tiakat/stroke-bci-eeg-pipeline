#!/usr/bin/env python
"""Run the SAME analysis on every dataset -> results/cross_dataset_*.csv

    python scripts/run_cross_dataset.py                 # all datasets, 2 workers (~40 min)
    python scripts/run_cross_dataset.py --datasets bnci2019 --n-jobs 1

Datasets (see src/kandu_eeg/datasets.py): liu2024, bnci2019, physionet_exec, physionet_imag, bnci2014.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import mne  # noqa: E402

from kandu_eeg import benchmark as B  # noqa: E402
from kandu_eeg import config as C  # noqa: E402
from kandu_eeg import datasets as DS  # noqa: E402

mne.set_log_level("ERROR")


def one(key, subject):
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    t0 = time.time()
    try:
        rec = DS.LOADERS[key][0](subject)
        rec.dataset = rec.dataset  # keep the loader's key
        out = B.analyse(rec)
        for r in out["rows"] + out["erd"]:
            r["dataset_key"] = key
        out["qc"]["dataset_key"] = key
        print(f"{key} {subject} ok {time.time() - t0:.0f}s", flush=True)
        return out
    except Exception:
        print(f"{key} {subject} FAILED\n{traceback.format_exc()}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["bnci2019", "bnci2014", "physionet_exec", "physionet_imag", "liu2024"])
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(k, s) for k in a.datasets for s in DS.LOADERS[k][1]]
    res = [r for r in Parallel(n_jobs=a.n_jobs)(delayed(one)(k, s) for k, s in jobs) if r]
    R = C.RESULTS_DIR
    pd.DataFrame([x for r in res for x in r["rows"]]).to_csv(R / "cross_dataset_auc.csv", index=False)
    pd.DataFrame([x for r in res for x in r["erd"]]).to_csv(R / "cross_dataset_erd.csv", index=False)
    pd.DataFrame([r["qc"] for r in res]).to_csv(R / "cross_dataset_qc.csv", index=False)
    print("saved", len(res), "recordings")


if __name__ == "__main__":
    main()
