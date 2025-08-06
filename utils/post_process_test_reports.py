import argparse
import json
import subprocess
from pathlib import Path


def simplify_gpu_name(gpu_name: str, simplified_names: list[str]) -> str:
    matches = []
    for simplified_name in simplified_names:
        if simplified_name in gpu_name:
            matches.append(simplified_name)
    if len(matches) == 1:
        return matches[0]
    return gpu_name

def parse_short_summary_line(line: str) -> tuple[str, int]:
    if line.startswith("PASSED"):
        return "passed", 1
    if line.startswith("FAILED"):
        return "failed", 1
    if line.startswith("SKIPPED"):
        line = line.split("[", maxsplit=1)[1]
        line = line.split("]", maxsplit=1)[0]
        return "skipped", int(line)
    if line.startswith("ERROR"):
        return "error", 1
    return None, 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Post process models test reports.")
    parser.add_argument("--path", "-p", help="Path to the reports folder")
    parser.add_argument("--gpu-name", "-g", help="GPU name", default=None)
    parser.add_argument("--commit-hash", "-c", help="Commit hash", default=None)

    args = parser.parse_args()
    path = Path(args.path)
    assert path.is_dir(), f"Path {path} is not a directory"

    # Get GPU name if available
    if args.gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name()
            gpu_name = gpu_name.replace(" ", "_").lower()
            gpu_name = simplify_gpu_name(gpu_name, ["mi300", "mi355", "h100", "a10"])
        except Exception as e:
            print(f"Failed to get GPU name with {e}")
            gpu_name = "unknown"
    else:
        gpu_name = args.gpu_name
    print(f"GPU: {gpu_name}")

    # Get commit hash if available
    if args.commit_hash is None:
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            commit_hash = commit_hash[:7]
        except Exception as e:
            print(f"Failed to get commit hash with {e}")
            commit_hash = "unknown"
    else:
        commit_hash = args.commit_hash
    print(f"Commit hash: {commit_hash}")

    # Initialize accumulators for collated report
    total_status_count = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        None: 0,
    }
    collated_report_buffer = []

    for model_dir in sorted(path.iterdir()):
        # Create a new entry for the model
        model_name = model_dir.name.removesuffix("_test_reports")
        collated_report_buffer.append(model_name)

        # Read short summary
        with open(model_dir / "summary_short.txt", "r") as f:
            short_summary_lines = f.readlines()

        # Parse short summary
        for line in short_summary_lines[1:]:
            collated_report_buffer.append(line.strip())
            status, count = parse_short_summary_line(line)
            total_status_count[status] += count

        # Add a separator between models
        # collated_report_buffer.append("\n")

    # Write collated report
    with open(f"collated_reports_{gpu_name}_{commit_hash}.txt", "w") as f:
        f.write(json.dumps({
            "gpu_name": gpu_name,
            "commit_hash": commit_hash,
            "total_status_count": total_status_count,
        }))
        f.write("\n\n")
        f.write("\n".join(collated_report_buffer))
