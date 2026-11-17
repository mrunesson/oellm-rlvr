#!/usr/bin/env python3
"""Write context-hash shard files for leonardo_laion_sif_build_rootless.sbatch (CONTEXT_HASH_FILE).

Uses the Harbor fork's own discovery (one entry per unique Docker build context) and skips images that are already
current in the SIF cache, so rerunning after a proxy expiry shards only the remaining work. Run on a login node with
the Harbor venv:
  PYTHONPATH=$HARBOR_LAION_ROOT/src $HARBOR_LAION_VENV/bin/python scripts/laion_context_hash_shards.py \
      $LAION_ROOT/<dataset>/tasks $SIF_CACHE <out-dir> [--shards 4]
Writes <out-dir>/shard-00 ... (round-robin over the sorted hashes) and prints the counts."""
import argparse
from pathlib import Path

from harbor.environments.apptainer.prebuild_sifs import _cached_build_is_current, discover_builds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", type=Path)
    p.add_argument("sif_cache", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--include-cached", action="store_true", help="also list images that are already built")
    a = p.parse_args()

    builds = discover_builds(a.tasks, a.sif_cache.expanduser().resolve())
    # Same defaults as the sbatch job: no base-image overrides, no task images, no wheelhouse.
    todo = [b for b in builds if a.include_cached or not _cached_build_is_current(
        b, base_image_overrides=None, prefer_task_images=False, wheel_dir=None)]
    print(f"{sum(len(b.task_names) for b in builds)} tasks, {len(builds)} unique builds, "
          f"{len(builds) - len(todo)} current in cache, {len(todo)} to build")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for old in a.out_dir.glob("shard-*"):
        old.unlink()
    n = max(1, min(a.shards, len(todo)))
    for i in range(n if todo else 0):
        hashes = [b.context_hash for b in todo[i::n]]
        path = a.out_dir / f"shard-{i:02d}"
        path.write_text("\n".join(hashes) + "\n")
        print(f"{path}: {len(hashes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
