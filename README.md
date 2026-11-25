## Results

Two measurements, because they answer different questions.

**Synthetic held-out test** (`test.psv`, 157 rows, same generator as training):

| Metric | Value |
|---|---|
| Junk recall | 93.4% |
| False dismissals | 0 of 66 |
| Accuracy | 96.2% |

This number is optimistic by construction: the test rows were written by the
same process as the training rows, so they share vocabulary. Do not quote it as
real-world accuracy.

**Real device notifications** (grouped split, whole title-families held out):

| Metric | Value |
|---|---|
| Junk recall | **86.3%** |
| False dismissals | **2 of 48** |
| Accuracy | **89.5%** |

Measured on 392 notifications captured on a real phone, with the split made by
title-family so paraphrases cannot appear on both sides. Those raw captures are
**not published** — they are the user's personal messages — but the split, the
scoring code (`tools/eval_export.py`) and the numbers are.

The gap between 93.4% and 86.3% is the honest cost of leaving the synthetic
distribution.


## Layout

```
data/          labelled corpus, pipe separated: app|title|text|JUNK|KEEP
model/         exported weights (model.tsv.gz)
tools/train.py training, threshold sweep, export
metrics.json   what the last run produced
```

Data splits are by **template family**, not by row: a wording family never straddles
train and test, so the score is not inflated by near-duplicate paraphrases.

## Model format

Gzipped, line based, three tab-separated fields:

```
#bias	-0.412000
free	1.405	0.8213
50% off	2.108	0.7741
```

At inference:

```
score = bias + Σ (tf × idf × coefficient)
dismiss when score > 0
```

The decision threshold is folded into the bias term, so the app needs no threshold of
its own. Tokens are matched case-insensitively on word uni- and bigrams.

## Train

```bash
python3 -m pip install scikit-learn numpy
python3 tools/train.py
```

Writes `model/model.tsv.gz` and `metrics.json`. Deterministic given the same corpus.

## Use it in an app

Drop `model/model.tsv.gz` into `assets/`, or publish it so installed apps can update
without a release — the app polls a small manifest weekly:

```json
{
  "version": 6,
  "url": "https://raw.githubusercontent.com/<owner>/<repo>/main/model/model.tsv.gz",
  "sha256": "<sha256 of the file>",
  "minApp": 58,
  "notes": "what changed"
}
```

The client verifies the checksum, parses the file, then runs three probes (an OTP must
stay, a known promo must go, a personal message must stay) before activating it. Any
failure keeps the previous model.

## Corpus

Every row here is **synthetic or written for this project** — Indian apps, US apps,
system updates, insurance and credit-card advertising, job-portal mail, and the
conversational messages that a naive model destroys. There is no data collected from
any user's phone, and no personal information.

| File | Rows | Purpose |
|---|---|---|
| `train.psv` | 757 | main five-class corpus |
| `existing_md.psv` | 83 | real-world cases harvested by hand |
| `extra_keep.psv` | 20 | greetings, dev-tool and system messages |
| `extra_tricky.psv` | 61 | smart-inbox ads, digests, profile nudges |
| `extra_apps.psv` | 41 | delivery and travel apps |
| `extra_apps2.psv` | 41 | banking, jobs, social |
| `extra_finance.psv` | 33 |
| `extra_comms.psv` | 22 |
| `extra_dev.psv` | 27 | insurance and card *advertising* vs real bank alerts |
| `dev.psv` | 148 | threshold selection |
| `test.psv` | 157 | reported numbers only |

## Limitations

- **Bag of words, no syntax.** It cannot tell a recruiter invite that ends
  "Apply now" from a mass ad that says the same, unless both appear in training.
- **~20 rows per sender** in the real captures, so per-sender behaviour is
  weakly learned. A sender-weighted feature was tried and measured: no gain.
- **~40% of real decisions are marginal** (within ±0.6 of the threshold).
- **English / Indian English only.** No Hinglish, no other scripts.

## Licence

MIT. Use it, fork it, retrain it on your own notifications — that is what it is for.
