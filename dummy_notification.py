import os
import json


REPORTS_DIR = "reports/multi-gpu_models_models"


Out = {}

for model in os.listdir(REPORTS_DIR):

    model_key = "models_" + model.removesuffix("_test_reports")
    Out[model_key] = {
        "errors": 0,
        "success": 0,
        "skipped": 0,

        "time_spent": ["0:03:23", "0:03:02"],
    }

    short_summary_file = os.path.join(REPORTS_DIR, model, "summary_short.txt")
    with open(short_summary_file, "r") as f:
        lines = f.readlines()

    failures = []
    lines = lines[1:] # skip the header
    for line in lines:
        if line.startswith("FAILED "):
            line = line.removeprefix("FAILED ")
            line, trace = line.split(" - ", 1)
            failures.append({
                "line": line,
                "trace": trace,
            })
        elif line.startswith("PASSED "):
            Out[model_key]["success"] += 1
        elif line.startswith("SKIPPED "):
            line = line.removeprefix("SKIPPED [")
            line = line.split("] ", 1)[0]
            Out[model_key]["skipped"] += int(line)
        elif line.startswith("ERROR "):
            Out[model_key]["errors"] += 1

    failed_keys = ["PyTorch", "TensorFlow", "Flax", "Tokenizers", "Pipelines", "Trainer", "ONNX", "Auto", "Quantization", "Unclassified"]

    Out[model_key]["failed"] = {}
    for k in failed_keys:
        Out[model_key]["failed"][k] = {"unclassified": 0, "single": 0, "multi": 0}

    Out[model_key]["failed"]["PyTorch"]["single"] = len(failures)
    Out[model_key]["failures"] = {
        "single": failures,
        "multi": failures,
    }
    Out[model_key]["job_link"] = {
        "single": "",
        "multi": "",
    }

models = list(Out.keys())
models.sort()

sorted_Out = {}
for model in models:
    sorted_Out[model] = Out[model]

with open("outi.json", "w") as f:
    json.dump(sorted_Out, f, indent=4)
