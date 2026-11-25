# Notification classifier

An on-device text classifier that decides whether an Android notification is **junk**
(promotions, engagement bait, digests) or something worth **keeping** (OTP, payments,
orders, security alerts, messages from people).

It is the model behind [AI Android Notification Filter](https://github.com/shreyash-sharma-byte/ai-android-notification-filter):
plain TF-IDF plus logistic regression, ~76 KB, no runtime dependencies, no network at
inference time. It runs in a few milliseconds on a mid-range phone.

## Results

Measured on a held-out split that no setting was tuned against:

| Metric | Value |
|---|---|
| Junk recall | **93.4%** |
| False dismissals | **0 of 66** |
| Accuracy | **96.2%** |
| Training rows | 1,027 |
| Features | 7,584 |
| Decision threshold | 0.10 (chosen on `dev`, zero false dismissals) |

The number that matters most is the second one. A notification filter that removes a
promotion is mildly useful; one that removes an OTP, a payment alert or a message from
your mother is actively harmful. The threshold is chosen so that `dev` has **no** false
dismissals, and everything else is traded away for that.

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
| `extra_finance.psv` | 33 | insurance and card *advertising* vs real bank alerts |
| `dev.psv` | 148 | threshold selection |
| `test.psv` | 157 | reported numbers only |

## Licence

MIT. Use it, fork it, retrain it on your own notifications — that is what it is for.
