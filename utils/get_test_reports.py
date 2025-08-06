import argparse
import contextlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch


# Mapping from suite name to test directory under `tests/`
SUITE_TO_PATH = {
    "run_models_gpu": "models",
    "run_pipelines_torch_gpu": "pipelines",
    "run_examples_gpu": "examples",
    "run_torch_cuda_extensions_gpu": "utils/torch_cuda_extensions",
}

IMPORTANT_MODELS = ["auto", "bert", "clip", "t5", "xlm-roberta", "wav2vec2", "llama", "opt", "longformer", "vit",
                    "whisper", "gemma3", "gpt2", "qwen2"]


def is_valid_test_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith("__") and not path.name.startswith(".")


def run_pytest(
    suite: str,
    subdir: Path,
    root_test_dir: Path,
    machine_type: str,
    dry_run: bool,
    tmp_cache: str,
    device_only: bool
):
    relative_path = subdir.relative_to(root_test_dir)
    report_name = f"{machine_type}_{suite}_{relative_path}_test_reports"
    print(f"Suite: {suite} | Running on: {relative_path}")

    cmd = ["python3", "-m", "pytest", "-rsfE", "-v", f"--make-reports={report_name}", str(subdir)]
    if device_only:
        cmd = cmd + ["-m", "not not_device_test"]

    ctx_manager = tempfile.TemporaryDirectory(prefix=tmp_cache) if tmp_cache else contextlib.nullcontext()
    with ctx_manager as tmp_dir:

        env = os.environ.copy()
        if tmp_cache:
            env["HUGGINGFACE_HUB_CACHE"] = tmp_dir

            print(f"Using temporary cache located at {tmp_dir = }")

        print("Command:", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=False, env=env)

def handle_suite(
    suite: str,
    test_root: Path,
    machine_type: str,
    dry_run: bool,
    tmp_cache: str = "",
    resume_at: Optional[str] = None,
    only_in: Optional[list[str]] = None,
    device_only: bool = True,
    process_id: int = 1,
    total_processes: int = 1,
) -> None:
    # Check suite
    if suite not in SUITE_TO_PATH:
        print(f"Unknown suite: {suite}")
        return None
    # Check path to suite
    subpath = SUITE_TO_PATH[suite]
    full_path = test_root / subpath
    if not full_path.exists():
        print(f"Test folder does not exist: {full_path}")
        return

    # Establish the list of subdir to go through
    subdirs = sorted(full_path.iterdir())
    subdirs = [s for s in subdirs if is_valid_test_dir(s)]
    if resume_at is not None:
        subdirs = [s for s in subdirs if s.name >= resume_at]
    if only_in is not None:
        subdirs = [s for s in subdirs if s.name in only_in]
    if subdirs and total_processes > 1:
        subdirs = subdirs[process_id::total_processes]

    # If the subdir list is not empty, go through each
    if subdirs:
        for subdir in subdirs:
            run_pytest(suite, subdir, test_root, machine_type, dry_run, tmp_cache, device_only)
    # Otherwise, launch pytest from the full path
    else:
        run_pytest(suite, full_path, test_root, machine_type, dry_run, tmp_cache, device_only)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run selected test suites recursively.")
    parser.add_argument("folder", help="Path to test root folder (e.g., ./tests)")

    parser.add_argument("--suites", nargs="+", required=True, help="List of test suite names to run")
    parser.add_argument("--device-only", action="store_false", help="Only run tests dependant on the device")
    parser.add_argument("--machine-type", type=str, default="", help="Machine type")

    parser.add_argument("--resume-at", type=str, default=None, help="Resume at a specific subdir")
    parser.add_argument("--only-in", type=str, nargs="+", help="Only run tests in the given subdirs")

    parser.add_argument(
        "--processes", type=int, nargs="+",
        help="Number of processes running the CI: format as process_id total_processes"
    )

    parser.add_argument("--run-slow", action="store_true", help="Run slow tests instead of skipping them")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without running them")
    parser.add_argument(
        "--tmp-cache", type=str, help="Change HUGGINGFACE_HUB_CACHE to a tmp dir for each test."
    )
    args = parser.parse_args()

    # Handle run slow
    if args.run_slow:
        os.environ["RUN_SLOW"] = "yes"
        print("[WARNING] Running slow tests.")
    else:
        print("[WARNING] Skipping slow tests.")

    # Handle multiple CI processes
    if args.processes is None:
        process_id, total_processes = 1, 1
    elif len(args.processes) == 2:
        process_id, total_processes = args.processes
    else:
        raise ValueError(f"Invalid processes argument: {args.processes}")

    # Assert test root exists
    test_root = Path(args.folder).resolve()
    if not test_root.exists():
        print(f"Root test folder not found: {test_root}")
        exit(1)

    # Infer machine type if not provided
    if args.machine_type == "":
        if not torch.cuda.is_available():
            machine_type = "cpu"
        else:
            machine_type = "multi-gpu" if torch.cuda.device_count() > 1 else "single-gpu"
    else:
        machine_type = args.machine_type

    # Reduce the scope for models if necessary
    only_in = args.only_in if args.only_in else None
    if only_in == ["IMPORTANT_MODELS"]:
        only_in = IMPORTANT_MODELS

    # Launch suites
    for suite in args.suites:
        handle_suite(
            suite=suite,
            test_root=test_root,
            machine_type=machine_type,
            dry_run=args.dry_run,
            tmp_cache=args.tmp_cache,
            resume_at=args.resume_at,
            only_in=only_in,
            device_only=args.device_only,
            process_id=process_id,
            total_processes=total_processes,
        )
