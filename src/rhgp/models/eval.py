from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from rhgp.models.baselines import always_a_proba, persistence_proba
from rhgp.models.train import feature_columns, time_split


def parse_thresholds(spec: str) -> list[float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    thresholds: list[float] = []
    for p in parts:
        v = float(p)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {v}")
        thresholds.append(v)
    if not thresholds:
        raise ValueError("No thresholds parsed")
    return thresholds


def evaluate_threshold(
    y_true: np.ndarray, p_fail: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (p_fail >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=cast(Any, 0),
    )
    return {
        "threshold": threshold,
        "precision_fail": float(p),
        "recall_fail": float(r),
        "f1_fail": float(f1),
    }


def format_threshold_table(rows: list[dict[str, float]]) -> str:
    header = "threshold\tprecision_fail\trecall_fail\tf1_fail"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['threshold']:.3f}\t{r['precision_fail']:.3f}\t{r['recall_fail']:.3f}\t{r['f1_fail']:.3f}"
        )
    return "\n".join(lines)


def dataset_fingerprint(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate model and baselines.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Key name used in metrics JSON (default: inferred from model filename).",
    )
    p.add_argument("--test-start", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated thresholds to print a FAIL precision/recall/F1 table.",
    )
    args = p.parse_args(argv)

    df = pd.read_parquet(args.data)
    _, test_df, cutoff, split_method, split_fraction = time_split(
        df, test_start=args.test_start
    )
    X_test = test_df[feature_columns()]
    y_test = test_df["y_t1"].astype(int).to_numpy()

    model_key = args.model_key
    if not model_key:
        name = args.model.stem.lower()
        if "logreg" in name:
            model_key = "logreg"
        elif "rf" in name or "randomforest" in name:
            model_key = "rf"
        else:
            model_key = "model"

    model = joblib.load(args.model)
    p_fail = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, object] = {"n_test": int(len(test_df))}
    metrics["run_metadata"] = {
        "model_key": model_key,
        "model_name": args.model.name,
        "split_method": split_method,
        "test_fraction": split_fraction,
        "cutoff_date": pd.to_datetime(cutoff).date().isoformat(),
        "dataset_fingerprint": dataset_fingerprint(args.data),
        "feature_columns": feature_columns(),
    }
    metrics[model_key] = evaluate_threshold(y_test, p_fail, threshold=args.threshold)

    p_fail_always_a = always_a_proba(test_df).to_numpy()
    metrics["always_a"] = evaluate_threshold(y_test, p_fail_always_a, threshold=args.threshold)

    p_fail_persist = persistence_proba(test_df).to_numpy()
    metrics["persistence"] = evaluate_threshold(y_test, p_fail_persist, threshold=args.threshold)

    if args.thresholds:
        thresholds = parse_thresholds(args.thresholds)
        rows = [evaluate_threshold(y_test, p_fail, threshold=t) for t in thresholds]
        best = max(rows, key=lambda r: r["f1_fail"])
        metrics[f"{model_key}_threshold_tuning"] = {
            "thresholds": thresholds,
            "rows": rows,
            "best": best,
            "chosen_threshold": float(args.threshold),
        }
        print(format_threshold_table(rows))

    y_pred = (p_fail >= args.threshold).astype(int)
    metrics[f"{model_key}_report"] = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=cast(Any, 0),
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(["true_ok(A)", "true_fail(BC+)"]),
        columns=pd.Index(["pred_ok(A)", "pred_fail(BC+)"]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cm_df.to_csv(args.out_dir / "confusion_matrix.csv", index=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
