#!/usr/bin/env python3
"""Summarise SkyRL's dumped evaluations (exports/dumped_evals/global_step_N_evals/*.jsonl) for single-turn tasks
scored on a "#### <answer>" line, such as GSM8K. Plain python3 (3.6+), no packages needed, so it runs on a login node.

    python3 scripts/skyrl_eval_summary.py RUN_ROOT                 # one row per evaluated step
    python3 scripts/skyrl_eval_summary.py RUN_ROOT --csv out.csv   # the same table as CSV
    python3 scripts/skyrl_eval_summary.py RUN_ROOT --question 3    # one test question at every step
    python3 scripts/skyrl_eval_summary.py RUN_ROOT --wrong 116     # the wrong answers at step 116

RUN_ROOT is the run directory ($OELLM_SCRATCH/oellm-rl/outputs/skyrl/<RUN_NAME>) or its dumped_evals folder.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

COLUMNS = ["step", "n", "accuracy", "correct", "has_marker", "wrong_has_marker", "no_marker", "cut_off",
           "cut_off_correct", "avg_tokens", "ends_eos", "ends_im_end"]


def eval_dir(root):
    for candidate in (root, os.path.join(root, "dumped_evals"), os.path.join(root, "exports", "dumped_evals")):
        if glob.glob(os.path.join(candidate, "global_step_*_evals")):
            return candidate
    sys.exit("no global_step_*_evals folders under " + root)


def steps(folder):
    found = {}
    for path in glob.glob(os.path.join(folder, "global_step_*_evals")):
        match = re.search(r"global_step_(\d+)_evals$", path)
        if match:
            found[int(match.group(1))] = path
    return [(step, found[step]) for step in sorted(found)]


def rows_of(path):
    rows = []
    for name in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        if os.path.basename(name) == "aggregated_results.jsonl":
            continue
        with open(name) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summary(step, path):
    rows = rows_of(path)
    out = [r["output_response"].rstrip() for r in rows]
    correct = [r["score"] > 0 for r in rows]
    answered = ["####" in o for o in out]
    cut = [r.get("stop_reason") == "length" for r in rows]
    avg_tokens = ""
    aggregated = os.path.join(path, "aggregated_results.jsonl")
    if os.path.exists(aggregated):
        with open(aggregated) as handle:
            avg_tokens = round(json.loads(handle.readline()).get("eval/all/generate/avg_num_tokens", 0))
    n = len(rows)
    return {
        "step": step, "n": n, "accuracy": round(sum(correct) / n, 4) if n else 0, "correct": sum(correct),
        "has_marker": sum(answered), "wrong_has_marker": sum(a and not c for a, c in zip(answered, correct)),
        "no_marker": n - sum(answered), "cut_off": sum(cut),
        "cut_off_correct": sum(k and c for k, c in zip(cut, correct)), "avg_tokens": avg_tokens,
        "ends_eos": sum(o.endswith("<eos>") for o in out), "ends_im_end": sum(o.endswith("<|im_end|>") for o in out),
    }


def index_of(row):
    return row.get("env_extras", {}).get("extra_info", {}).get("index")


def truth_of(row):
    return row.get("env_extras", {}).get("reward_spec", {}).get("ground_truth")


def main():
    parser = argparse.ArgumentParser(description="Summarise SkyRL dumped evaluations.")
    parser.add_argument("root")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    parser.add_argument("--question", type=int, help="show this test question (extra_info.index) at every step")
    parser.add_argument("--wrong", type=int, metavar="STEP", help="show the wrong answers at this step")
    parser.add_argument("--limit", type=int, default=10, help="how many wrong answers to show (default 10)")
    parser.add_argument("--chars", type=int, default=600, help="characters of each response to show (default 600)")
    args = parser.parse_args()
    found = steps(eval_dir(args.root))

    if args.question is not None:
        for step, path in found:
            for row in rows_of(path):
                if index_of(row) == args.question:
                    print("=== step %d  score=%s  stop=%s  truth=%s" % (step, row["score"], row.get("stop_reason"),
                                                                       truth_of(row)))
                    print(row["output_response"][-args.chars:].rstrip() + "\n")
        return

    if args.wrong is not None:
        paths = dict(found)
        if args.wrong not in paths:
            sys.exit("no evaluation at step %d; steps: %s" % (args.wrong, " ".join(str(s) for s, _ in found)))
        wrong = [r for r in rows_of(paths[args.wrong]) if r["score"] <= 0]
        print("%d wrong answers at step %d; showing %d\n" % (len(wrong), args.wrong, min(args.limit, len(wrong))))
        for row in wrong[:args.limit]:
            question = row.get("env_extras", {}).get("extra_info", {}).get("question", "")
            print("=== question %s  truth=%s  stop=%s" % (index_of(row), truth_of(row), row.get("stop_reason")))
            print("Q: " + question)
            print("A: ..." + row["output_response"][-args.chars:].rstrip() + "\n")
        return

    table = [summary(step, path) for step, path in found]
    widths = [max(len(c), *(len(str(t[c])) for t in table)) for c in COLUMNS]
    print("  ".join(c.rjust(w) for c, w in zip(COLUMNS, widths)))
    for t in table:
        print("  ".join(str(t[c]).rjust(w) for c, w in zip(COLUMNS, widths)))
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(table)
        print("wrote " + args.csv)


if __name__ == "__main__":
    main()
