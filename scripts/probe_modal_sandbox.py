#!/usr/bin/env python3
"""Start one Modal sandbox from a Harbor task's environment/Dockerfile, run a command in it, and terminate it.

Proves the path a Harbor ``modal`` trial takes: credentials, image build (cached on Modal afterwards, so the first
training step does not wait for it), sandbox start, exec, and the sandbox's own internet access (the CodeContests
verifier installs packages). With OELLM_MODAL_PROXY set, Modal goes through that proxy (see oellm_rlvr.modal_proxy),
as it does in the training job.

  probe_modal_sandbox.py --task-dir <harbor task dir> [--app-name __harbor__] [--output probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CHECK = "python3 --version && (getent hosts astral.sh >/dev/null && echo internet=ok || echo internet=missing)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--app-name", default="__harbor__", help="Harbor's default Modal app name")
    parser.add_argument("--timeout", type=int, default=900, help="sandbox lifetime in seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from oellm_rlvr.modal_proxy import install_modal_proxy

    proxy = install_modal_proxy()
    import modal

    environment = args.task_dir / "environment"
    dockerfile = environment / "Dockerfile"
    if not dockerfile.is_file():
        raise SystemExit(f"no Dockerfile at {dockerfile}")
    report: dict = {"task": str(args.task_dir), "proxy": proxy, "modal": modal.__version__, "ok": False}
    start = time.monotonic()
    sandbox = None
    try:
        app = modal.App.lookup(args.app_name, create_if_missing=True)
        image = modal.Image.from_dockerfile(str(dockerfile), context_dir=str(environment))
        sandbox = modal.Sandbox.create(app=app, image=image, timeout=args.timeout, cpu=1, memory=1024)
        report["sandbox_id"] = sandbox.object_id
        report["start_seconds"] = round(time.monotonic() - start, 1)
        process = sandbox.exec("bash", "-c", CHECK, timeout=120)
        process.wait()
        report["stdout"] = process.stdout.read().strip()
        report["stderr"] = process.stderr.read().strip()
        report["returncode"] = process.returncode
        report["ok"] = process.returncode == 0 and "internet=ok" in report["stdout"]
    except Exception as error:  # noqa: BLE001 - report any failure, then exit 1
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate()
            except Exception as error:  # noqa: BLE001
                report["terminate_error"] = f"{type(error).__name__}: {error}"
    report["total_seconds"] = round(time.monotonic() - start, 1)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
