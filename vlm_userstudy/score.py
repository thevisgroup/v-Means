# -*- coding: utf-8 -*-
"""Score VLM responses with the same keys as the human study.

Accuracy denominators include valid answers only. Format-validity rates are
reported separately so malformed or missing model output remains visible.

Q6 standards:
  human12 (primary): 12 items; Finding empty space excluded (None);
                     distractors and Early termination scored No.
                     Used for the human-comparable analysis.
  design13 (sensitivity): all 13 presented items; only the four distractors
                          scored No; Early termination and Finding empty space
                          both scored Yes.

Usage:  python score.py [outputs/vlm_responses.csv]
"""

import csv
import sys
from collections import defaultdict

import config as C
import questionnaire as Q

YES, NO = Q.Q6_COLUMNS


def q6_key(step, standard):
    if standard == "design13":
        return NO if step in Q.DISTRACTORS else YES
    if standard == "human12":
        if step in Q.DISTRACTORS or step == "Early termination":
            return NO
        if step == "Finding empty space":
            return None  # excluded from human-comparable 12-item scoring
        return YES
    raise ValueError(standard)


def _parse_int(value, low, high):
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if low <= parsed <= high else None


def _video_format_valid(row, vid):
    scalar_valid = (
        row.get(f"{vid}_Q1") in Q.YES_NO
        and row.get(f"{vid}_Q2") in Q.LIKERT_5
        and row.get(f"{vid}_Q3") in Q.CLUSTER_COUNTS
        and row.get(f"{vid}_Q4") in Q.TRUE_FALSE
        and _parse_int(row.get(f"{vid}_Q5"), 0, 100) is not None
        and row.get(f"{vid}_Q7") in Q.YES_NO
    )
    grid_valid = all(
        row.get(f"{vid}_Q6 [{step}]") in Q.Q6_COLUMNS
        for step in Q.ALGORITHM_STEPS
    )
    return scalar_valid and grid_valid


def _overall_format_valid(row):
    return all(_parse_int(row.get(k), 1, 5) is not None
               for k in ["Q17", "Q18", "Q19", "Q20"])


def score_rows(path):
    per_model = defaultdict(lambda: {
        "n": 0,
        "video_format_correct": 0, "video_format_total": 0,
        "overall_format_correct": 0, "overall_format_total": 0,
        "q3_correct": 0, "q3_total": 0,
        "q4_correct": 0, "q4_total": 0,
        "q6_d13_correct": 0, "q6_d13_total": 0,
        "q6_h12_correct": 0, "q6_h12_total": 0,
        "q5_sum": 0, "q5_n": 0,
    })
    q4_correct_opt = Q.PER_VIDEO_QUESTIONS["Q4"]["correct"]

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = per_model[row["model_tag"]]
            s["n"] += 1
            s["overall_format_total"] += 1
            if _overall_format_valid(row):
                s["overall_format_correct"] += 1

            for v in C.VIDEOS:
                vid = v["id"]
                s["video_format_total"] += 1
                if _video_format_valid(row, vid):
                    s["video_format_correct"] += 1

                q3 = row.get(f"{vid}_Q3", "")
                if q3 in Q.CLUSTER_COUNTS:
                    s["q3_total"] += 1
                    if q3 == v["q3_expected"]:
                        s["q3_correct"] += 1

                q4 = row.get(f"{vid}_Q4", "")
                if q4 in Q.TRUE_FALSE:
                    s["q4_total"] += 1
                    if q4 == q4_correct_opt:
                        s["q4_correct"] += 1

                q5 = _parse_int(row.get(f"{vid}_Q5", ""), 0, 100)
                if q5 is not None:
                    s["q5_sum"] += q5
                    s["q5_n"] += 1

                for step in Q.ALGORITHM_STEPS:
                    ans = row.get(f"{vid}_Q6 [{step}]", "")
                    if ans not in Q.Q6_COLUMNS:
                        continue
                    key_d13 = q6_key(step, "design13")
                    s["q6_d13_total"] += 1
                    if ans == key_d13:
                        s["q6_d13_correct"] += 1
                    key_h12 = q6_key(step, "human12")
                    if key_h12 is not None:
                        s["q6_h12_total"] += 1
                        if ans == key_h12:
                            s["q6_h12_correct"] += 1
    return per_model


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "  n/a "


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else C.CSV_PATH
    per_model = score_rows(path)
    print("Q6 human12 is the primary human-comparable score (12 items, excl. "
          "Finding empty space); design13 is the sensitivity analysis (all 13 "
          "items). Accuracy uses valid-only denominators.\n")
    hdr = (
        f"{'model':20s} {'runs':>4s} {'Q3':>7s} {'Q3valid':>7s} "
        f"{'Q4':>7s} {'Q4valid':>7s} {'Q6(h12)':>8s} {'Q6(d13)':>8s} "
        f"{'Q6valid':>7s} {'meanQ5':>7s} {'videoFmt':>8s} {'overallFmt':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag, s in sorted(per_model.items()):
        video_n = s["n"] * len(C.VIDEOS)
        q6_possible = video_n * len(Q.ALGORITHM_STEPS)
        q5 = f"{s['q5_sum'] / s['q5_n']:7.1f}" if s["q5_n"] else "    n/a"
        print(
            f"{tag:20s} {s['n']:4d} "
            f"{pct(s['q3_correct'], s['q3_total'])} "
            f"{pct(s['q3_total'], video_n)} "
            f"{pct(s['q4_correct'], s['q4_total'])} "
            f"{pct(s['q4_total'], video_n)} "
            f"{pct(s['q6_h12_correct'], s['q6_h12_total']):>8s} "
            f"{pct(s['q6_d13_correct'], s['q6_d13_total']):>8s} "
            f"{pct(s['q6_d13_total'], q6_possible)} {q5} "
            f"{pct(s['video_format_correct'], s['video_format_total']):>8s} "
            f"{pct(s['overall_format_correct'], s['overall_format_total']):>10s}"
        )


if __name__ == "__main__":
    main()
