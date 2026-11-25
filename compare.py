"""Assemble a comparison after verifying that the two models share holdout hours."""
import argparse
import hashlib
import json
from pathlib import Path

MODES = ("with-history", "without-history")


def load(directory, name):
    return json.loads((directory / f"{name}.json").read_text())


def make_comparison(web):
    reports, specs, links, examples = [], [], [], []
    for mode in MODES:
        folder = web / "models" / mode
        report = load(folder, "metrics")
        spec = load(folder, "model-spec")
        if report["history_mode"] != mode:
            raise ValueError(f"Unexpected experiment in {folder}")
        digest = hashlib.sha256((folder / "model.onnx").read_bytes()).hexdigest()
        if digest != report["model_sha256"]:
            raise ValueError(f"ONNX model does not match metrics in {folder}")
        reports.append(report)
        specs.append(spec)
        links.append({"id": mode, "label": report["label"], "model_sha256": digest})
        examples.append(load(folder, "examples"))
    full, reduced = reports
    for key in ("split", "test_hours_sha256", "wet_threshold_mm"):
        if full[key] != reduced[key]:
            raise ValueError(f"Comparison requires the same {key}")
    if list(specs[0]["references"]) != list(specs[1]["references"]):
        raise ValueError("Output reference order differs")

    restricted = {"historical_hrrr_mm", "recent_error_mm"}
    full_names, reduced_names = [[f["name"] for f in spec["features"]] for spec in specs]
    if not restricted.issubset(full_names) or [n for n in full_names if n not in restricted] != reduced_names:
        raise ValueError("Model history inputs violate the ablation specification")
    shared = [full_names.index(name) for name in reduced_names]
    if len(examples[0]) != len(examples[1]):
        raise ValueError("Model example counts differ")
    for a, b in zip(*examples):
        for key in ("valid_utc", "hrrr_mm", "reference_mm"):
            if a[key] != b[key]:
                raise ValueError(f"Example {key} differs")
        if [[row[i] for i in shared] for row in a["history"]] != b["history"]:
            raise ValueError("Shared weather values differ")
    table = {}
    for key, scores in full["metrics"].items():
        table[key] = {}
        for model in ("Raw HRRR", "Scaled HRRR"):
            if scores[model] != reduced["metrics"][key][model]:
                raise ValueError(f"The common {model} baseline differs")
            table[key][model] = scores[model]
        for report in reports:
            table[key][report["label"]] = report["metrics"][key]["LSTM · six hours"]
    comparison = {"models": links, "split": full["split"], "wet_threshold_mm": full["wet_threshold_mm"],
                  "test_hours_sha256": full["test_hours_sha256"], "metrics": table}
    (web / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, default=Path("web"))
    make_comparison(parser.parse_args().web)
