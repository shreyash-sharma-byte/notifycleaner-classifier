#!/usr/bin/env python3
"""Audit a NotifyCleaner sample export.

    python3 tools/eval_export.py samples.tsv [--verbose]

Columns: ts, app, title, text, pkg, verdict, score. The verdict and score are
the app's own output, so this is not ground truth: the script applies a written
policy to each row and reports where the app disagrees with it.
"""
import argparse
import collections
import re
import sys

JUNK_IF = [r"new jobs? for you", r"jobs? matching", r"apply now", r"apply fast",
           r"closing soon", r"masterclass|webinar|workshop", r"top candidate|top applicant",
           r"viewed your profile|profile views|search appearances", r"missed notifications",
           r"reacted to a post|reposted|completed zip|suggested for you",
           r"hiring\. view more|is hiring", r"\d+ new (notifications|updates|jobs)",
           r"catch up with them|you may have missed", r"free to play",
           r"% off|cashback|prizes? worth|reward points",
           r"insurance|term life|home loan|personal loan|credit card|loan up to",
           r"commission|terms updated|invite friends|refer and earn",
           r"follow [a-z].*\(", r"^follow back$", r"try .*premium|upgrade to (pro|premium)",
           r"weekly digest|what happened in your", r"release v?\d|new features",
           r"trending|searches included you", r"practice these.*questions",
           r"unread messages|started a huddle|nitro", r"^screenshot saved$",
           r"see full forecast", r"^\d+$", r"turn on power saving"]

KEEP_IF = [r"otp|one[- ]time password|verification code",
           r"debited|credited|paid|payment (of|successful|received)",
           r"order (is|has been|delivered)|out for delivery|arriving",
           r"pnr|ticket|platform", r"missed call|is calling|calling you",
           r"shortlisted|interview|recruiter considers|job nvite|offer letter",
           r"workflow ci failed|pipeline.*failed|deploy.*failed|build failed",
           r"needs your review|assigned to you", r"security alert|vulnerability|incident",
           r"due on|bill of", r"^\[amount\]$", r"meeting|starts in \d+ min|join now",
           r"are you free|call me when", r"daily activity|target reached",
           r"auto back ?up|legal terms"]


def judge(title, text, app_verdict):
    blob = f"{title} {text}".lower()
    for p in KEEP_IF:
        if re.search(p, blob):
            return "KEEP"
    for p in JUNK_IF:
        if re.search(p, blob):
            return "JUNK"
    return app_verdict


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            rows.append({"ts": f[0], "app": f[1], "title": f[2], "text": f[3],
                         "verdict": f[5], "score": f[6] if len(f) > 6 else ""})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    rows = load(a.path)
    if not rows:
        print("no rows parsed")
        return 1

    counts = collections.Counter(r["verdict"] for r in rows)
    bad = []
    for r in rows:
        want = judge(r["title"], r["text"], r["verdict"])
        if want != r["verdict"]:
            bad.append((r, want))

    print(f"rows {len(rows)}  JUNK {counts.get('JUNK', 0)}  KEEP {counts.get('KEEP', 0)}")
    print(f"policy disagreements: {len(bad)} ({len(bad) / len(rows) * 100:.1f}%)")
    by_app = collections.Counter(r["app"] for r, _ in bad)
    for app, n in by_app.most_common(6):
        print(f"  {n:3d}  {app}")
    if a.verbose:
        for r, want in bad[:40]:
            print(f"  app={r['verdict']:4s} policy={want:4s} [{r['app']}] {r['title'][:40]} | {r['text'][:50]}")

    risky = [r for r in rows if r["score"] not in ("", "0")
             and -60 <= int(r["score"]) <= 60]
    print(f"fragile band (|score| <= 0.60): {len(risky)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
