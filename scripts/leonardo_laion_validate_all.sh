#!/bin/bash
# Validate every task of a LAION dataset that has a successfully built image, in chunks of debug jobs, then merge the
# per-chunk results into the one list the rollout job reads. Run on a login node from the repo checkout.
#   scripts/leonardo_laion_validate_all.sh submit   # chunk the tasks and submit one leonardo_laion_validate.sbatch job per chunk
#   scripts/leonardo_laion_validate_all.sh retry    # submit chunks the queue refused (debug QoS: 4 jobs per user) or that ended without results
#   scripts/leonardo_laion_validate_all.sh merge    # after the jobs end: write $RESULTS_ROOT/<dataset>-leonardo-compatible-all.jsonl
# A task is listed only if its image has a .sif.build.json (written after a successful build). Tasks without an image are
# written to no-image.txt in the run folder. merge uses the newest run unless RUN=<run folder> is set.
# Env: DATASET (rts-1k-rootless-apptainer), CHUNK (150 tasks per job; 200 used the full 30 minutes), ATTEMPTS (3), QOS
# (boost_qos_dbg; empty = normal queue), TIME (00:30:00), WORKERS_PER_NODE (16).
# The debug QoS accepts 4 jobs per user and runs 2 at a time; the others wait with reason QOSMaxJobsPerUserLimit.
set -euo pipefail
: "${OELLM_SCRATCH:?set the OELLM_* variables first (see the runbook, section 2)}"
: "${DATASET:=rts-1k-rootless-apptainer}"
: "${CHUNK:=150}"
: "${ATTEMPTS:=3}"
: "${QOS=boost_qos_dbg}"
: "${TIME:=00:30:00}"
: "${WORKERS_PER_NODE:=16}"
ROOT=$OELLM_SCRATCH/oellm-rl
: "${SIF_CACHE:=$ROOT/sif-cache-x86_64}"
: "${RESULTS_ROOT:=$ROOT/laion-validation}"
PY=$ROOT/venvs/harbor-laion/bin/python
TASKS=$ROOT/data/laion-ot/$DATASET/tasks

submit() {
  [[ -d $TASKS ]] || { echo "no tasks at $TASKS" >&2; exit 1; }
  [[ -f scripts/leonardo_laion_validate.sbatch ]] || { echo "run from the repo checkout" >&2; exit 1; }
  local run=$RESULTS_ROOT/$DATASET-full-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$run" logs
  # Tasks named in the success manifests of the image cache, restricted to this dataset's task folders.
  "$PY" - "$SIF_CACHE" "$TASKS" "$run" "$CHUNK" <<'PY'
import json, sys
from pathlib import Path
cache, tasks, run, chunk = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
all_tasks = sorted(d.name for d in tasks.iterdir() if d.is_dir())
have = set()
for m in cache.glob("*.sif.build.json"):
    if m.with_name(m.name[: -len(".build.json")]).is_file():
        have.update(json.loads(m.read_text()).get("task_names", []))
ok = [t for t in all_tasks if t in have]
missing = [t for t in all_tasks if t not in have]
(run / "no-image.txt").write_text("".join(t + "\n" for t in missing))
for i in range(0, len(ok), chunk):
    (run / f"chunk-{i // chunk:02d}.txt").write_text("".join(t + "\n" for t in ok[i:i + chunk]))
print(f"{len(all_tasks)} tasks: {len(ok)} with an image, {len(missing)} without (see no-image.txt); "
      f"{(len(ok) + chunk - 1) // chunk} chunks of up to {chunk}")
PY
  touch "$run/jobs.txt"
  echo "run folder: $run"
  submit_chunks "$run"
}

newest_run() {
  local run=${RUN:-$(ls -dt "$RESULTS_ROOT/$DATASET"-full-* 2>/dev/null | head -1)}
  [[ -f $run/jobs.txt ]] || { echo "no run with jobs.txt under $RESULTS_ROOT (set RUN=...)" >&2; exit 1; }
  echo "$run"
}

# Submit every chunk that has no job yet, or whose job has left the queue without writing a report (timed out or failed).
# jobs.txt keeps one "<job> <chunk>" line per submission; the last line for a chunk is the current one.
submit_chunks() {
  local run=$1 f chunk job id n=0
  local qos=(); [[ -n $QOS ]] && qos=(--qos="$QOS")
  for f in "$run"/chunk-*.txt; do
    chunk=$(basename "$f")
    job=$(awk -v c="$chunk" '$2 == c {j = $1} END {print j}' "$run/jobs.txt")
    if [[ -n $job ]]; then
      [[ -f $RESULTS_ROOT/$DATASET-list-$job/validation-report.json ]] && continue
      [[ -n $(squeue -h -j "$job" 2>/dev/null) ]] && continue
      echo "$chunk: job $job ended without results, resubmitting"
    fi
    if ! id=$(DATASET=$DATASET SUBSET=list TASK_LIST=$f ATTEMPTS=$ATTEMPTS WORKERS_PER_NODE=$WORKERS_PER_NODE \
        sbatch --parsable -A "$OELLM_ACCOUNT" "${qos[@]}" --time="$TIME" --export=ALL --job-name=laion-validate \
        scripts/leonardo_laion_validate.sbatch); then
      echo "STOP: the queue refused $chunk (the debug queue accepts 4 jobs per user; this is expected with more chunks)."
      echo "When a job has finished, run: $0 retry"
      return 0
    fi
    echo "$id $chunk" | tee -a "$run/jobs.txt"; n=$((n + 1))
  done
  echo "submitted $n job(s); nothing left to submit"
  echo "when the jobs have ended: $0 merge"
}

retry() {
  [[ -f scripts/leonardo_laion_validate.sbatch ]] || { echo "run from the repo checkout" >&2; exit 1; }
  local run; run=$(newest_run)
  echo "run folder: $run"
  submit_chunks "$run"
}

merge() {
  local run; run=$(newest_run)
  echo "run folder: $run"
  "$PY" - "$run" "$RESULTS_ROOT" "$DATASET" <<'PY'
import json, sys
from pathlib import Path
run, results, dataset = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ok, failed, unfinished, tested = [], [], [], 0
current = {}  # chunk -> latest job (a chunk may have been resubmitted)
for line in (run / "jobs.txt").read_text().split("\n"):
    if line.strip():
        job, chunk = line.split()
        current[chunk] = job
for f in sorted(run.glob("chunk-*.txt")):
    if f.name not in current:
        unfinished.append(f"(never submitted) {f.name}")
for chunk, job in sorted(current.items()):
    out = results / f"{dataset}-list-{job}"
    report = out / "validation-report.json"
    if not report.is_file():
        unfinished.append(f"{job} {chunk}")
        continue
    r = json.loads(report.read_text())
    tested += r["expected_tasks"]
    failed += r["failures"]
    ok += [json.loads(l)["task_id"] for l in (out / "leonardo-compatible.jsonl").read_text().split("\n") if l.strip()]
    print(f"  job {job} ({chunk}): {r['compatible']} of {r['expected_tasks']} usable")
merged = results / f"{dataset}-leonardo-compatible-all.jsonl"
if unfinished:
    print("NOT merged; these chunks have no results yet (still queued or running, failed, or never submitted):")
    print("".join(f"  {u}\n" for u in unfinished), end="")
    print("Wait for running jobs, then run `retry` to submit the rest (logs: logs/<job>.laion-validate.out).")
    sys.exit(1)
ok = sorted(set(ok))
merged.write_text("".join(json.dumps({"task_id": t}) + "\n" for t in ok))
(run / "failures.json").write_text(json.dumps(failed, indent=2) + "\n")
no_image = sum(1 for l in (run / "no-image.txt").read_text().split("\n") if l.strip())
print(f"usable: {len(ok)} of {tested} tested tasks ({no_image} more had no image)")
print(f"wrote {merged}")
print(f"why the others failed: {run / 'failures.json'}")
PY
}

case ${1:-} in
  submit) submit ;;
  retry) retry ;;
  merge) merge ;;
  *) echo "usage: $0 submit|retry|merge" >&2; exit 2 ;;
esac
