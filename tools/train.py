#!/usr/bin/env python3
"""Train the notification classifier, sweep the threshold, export the weights.

    python3 tools/train.py            # writes model/model.tsv.gz + metrics.json

The exported file is line-based gzipped TSV, which is what the Android app reads:

    #bias   <float>
    <token> <idf>   <coefficient>

At runtime the app computes score = bias + sum(tf * idf * coefficient) over the
tokens it finds in the notification, and dismisses the notification when
score > 0. The decision threshold is folded into the bias, so the app needs no
threshold of its own.
"""
import gzip
import json
import pathlib
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "model"

TRAIN_FILES = [
    "train.psv",
    "existing_md.psv",
    "extra_keep.psv",
    "extra_tricky.psv",
    "extra_apps.psv",
    "extra_apps2.psv",
    "extra_finance.psv",
]


def load(path):
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        label = parts[3].strip().upper()
        if label not in ("JUNK", "KEEP"):
            continue
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), label))
    return rows


def doc(row):
    return " ".join(row[:3]).lower()


def metrics(y, score, thr):
    pred = (score > thr).astype(int)
    junk = y == 1
    keep = y == 0
    recall = float(pred[junk].mean()) if junk.any() else 0.0
    fd = int(((pred == 1) & keep).sum())
    acc = float((pred == y).mean())
    return recall, fd, acc


def main():
    train, dev, test = [], [], []
    for name in TRAIN_FILES:
        train += load(DATA / name)
    dev = load(DATA / "dev.psv")
    test = load(DATA / "test.psv")
    print(f"train {len(train)} | dev {len(dev)} | test {len(test)}")

    y = np.array([1 if r[3] == "JUNK" else 0 for r in train])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
    x = vec.fit_transform([doc(r) for r in train])
    clf = LogisticRegression(C=10.0, class_weight="balanced", max_iter=3000).fit(x, y)

    dev_y = np.array([1 if r[3] == "JUNK" else 0 for r in dev])
    dev_s = clf.decision_function(vec.transform([doc(r) for r in dev]))

    # Lowest threshold with no false dismissal on dev, floored at 0.
    chosen = None
    for thr in np.arange(0.0, 0.95, 0.05):
        recall, fd, _ = metrics(dev_y, dev_s, thr)
        if fd == 0:
            chosen = (float(thr), recall)
            break
    if chosen is None:
        chosen = (0.5, 0.0)
    thr = chosen[0]
    print(f"threshold {thr:.2f} (dev recall {chosen[1] * 100:.1f}% at zero false dismissals)")

    test_y = np.array([1 if r[3] == "JUNK" else 0 for r in test])
    test_s = clf.decision_function(vec.transform([doc(r) for r in test]))
    recall, fd, acc = metrics(test_y, test_s, thr)
    print(f"test: recall {recall * 100:.1f}% | false dismissals {fd} | accuracy {acc * 100:.1f}%")

    OUT.mkdir(exist_ok=True)
    coef, names, idf = clf.coef_[0], vec.get_feature_names_out(), vec.idf_
    with gzip.open(OUT / "model.tsv.gz", "wt", encoding="utf-8") as f:
        f.write(f"#bias\t{clf.intercept_[0] - thr:.6f}\n")
        kept = 0
        for name, i, c in zip(names, idf, coef):
            if abs(c) > 1e-6:
                f.write(f"{name}\t{i:.6f}\t{c:.6f}\n")
                kept += 1
    report = {
        "threshold": thr,
        "features": kept,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "test_rows": len(test),
        "test": {"recall_junk": round(recall, 4), "false_dismissals": fd, "accuracy": round(acc, 4)},
    }
    (ROOT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote model/model.tsv.gz ({kept} features) and metrics.json")


if __name__ == "__main__":
    sys.exit(main())
