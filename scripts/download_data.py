#!/usr/bin/env python
"""Download every dataset used in the analysis.

    python scripts/download_data.py                 # all datasets (~6 GB)
    python scripts/download_data.py --only liu2024  # one dataset

| key       | source                                                      | size    |
|-----------|-------------------------------------------------------------|---------|
| liu2024   | figshare 10.6084/m9.figshare.21679035 (raw .mat + clinical) | ~1.9 GB |
| bnci2019  | Zenodo record 2222268 (P01-P09 offline sessions, GDF)       | ~2.0 GB |
| physionet | PhysioNet EEGMMIDB, subjects 1-50, runs 3/4/7/8/11/12       | ~0.8 GB |
| bnci2014  | BCI Competition IV-2a via MOABB                              | ~0.7 GB |

Layout: data/raw_src, data/meta (liu2024) and data_ext/<dataset> (others).
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, EXT = ROOT / "data", ROOT / "data_ext"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print("downloading", dest.name)
    urllib.request.urlretrieve(url, dest)


def liu2024():
    meta = {"participants.tsv", "participants.json", "task-motor-imagery_events.tsv",
            "task-motor-imagery_channels.tsv", "task-motor-imagery_eeg.json"}
    files = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/21679035"))["files"]
    for f in files:
        if f["name"] in meta:
            fetch(f["download_url"], DATA / "meta" / f["name"].replace("task-motor-imagery_", ""))
    if not (DATA / "raw_src" / "sourcedata").exists():
        src = next(f for f in files if f["name"] == "sourcedata.zip")
        z = DATA / "raw_src" / "sourcedata.zip"
        fetch(src["download_url"], z)
        zipfile.ZipFile(z).extractall(DATA / "raw_src"); z.unlink()


def bnci2019():
    for p in range(1, 10):
        out = EXT / "bnci2019" / f"P{p:02d}"
        if out.exists():
            continue
        z = EXT / "bnci2019" / f"P{p:02d}.zip"
        fetch(f"https://zenodo.org/records/2222268/files/P{p:02d}.zip?download=1", z)
        zipfile.ZipFile(z).extractall(out); z.unlink()
    fetch("https://zenodo.org/records/2222268/files/data%20set%20description.pdf?download=1", EXT / "bnci2019" / "description.pdf")


def physionet():
    from mne.datasets import eegbci
    for s in range(1, 51):
        if s in (38, 88, 89, 92, 100, 104):   # known faulty recordings in EEGMMIDB
            continue
        eegbci.load_data(s, [1, 3, 4, 7, 8, 11, 12], path=str(EXT / "physionet"), update_path=False)


def bnci2014():
    import mne
    from moabb.datasets import BNCI2014_001
    (EXT / "moabb").mkdir(parents=True, exist_ok=True)
    for k in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH"):
        mne.set_config(k, str(EXT / "moabb"), set_env=True)
    BNCI2014_001().get_data(subjects=list(range(1, 10)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["liu2024", "bnci2019", "physionet", "bnci2014"])
    a = ap.parse_args()
    for name, fn in [("liu2024", liu2024), ("bnci2019", bnci2019), ("physionet", physionet), ("bnci2014", bnci2014)]:
        if a.only in (None, name):
            print("==", name); fn()
    print("done")
