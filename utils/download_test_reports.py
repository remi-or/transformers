import fnmatch
import os
import zipfile

from utils.get_ci_error_statistics import download_artifact, get_artifacts_links


DEFAULT_ARTIFACTS = [
    "ci_results_run_models_gpu",
    "ci_results_run_examples_gpu",
    "ci_results_run_pipelines_torch_gpu",
    "ci_results_run_torch_cuda_extensions_gpu",
]

def download_and_extract_artifacts(workflow_run_id: int, artifacts: list[str], output_dir: str, token: str) -> dict[str, dict[str, str]]:
    """
    Download specified artifacts from a workflow run, unzip them into the output directory,
    and return their contents as a dictionary.
    """
    artifacts_links = get_artifacts_links(worflow_run_id=workflow_run_id, token=token)
    print(artifacts_links)
    results = {}

    os.makedirs(output_dir, exist_ok=False)

    for artifact_name in artifacts_links:
        if fnmatch.fnmatch(artifact_name, artifacts):
            artifact_url = artifacts_links[artifact_name]
            if not artifact_url:
                continue

            print(artifact_name)
            zip_path = os.path.join(output_dir, f"{artifact_name}.zip")
            print(zip_path)
            download_artifact(
                artifact_name=artifact_name,
                artifact_url=artifact_url,
                output_dir=output_dir,
                token=token
            )

            # Extract ZIP and read contents
            if os.path.isfile(zip_path):
                if "_" not in artifact_name:
                    raise ValueError("artifact_name must contain at least one underscore")

                prefix = "multi-gpu_run_models_gpu_models_"
                leaf = artifact_name.replace("multi-gpu_run_models_gpu_models_", "")
                extract_dir = os.path.join(output_dir, prefix[:-1], leaf)
                os.makedirs(extract_dir, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(path=extract_dir)
                    os.remove(zip_path)
                except zipfile.BadZipFile:
                    print(f"Bad ZIP file: {zip_path}")
                except Exception as e:
                    print(f"Error extracting {zip_path}: {e}")
    return results
