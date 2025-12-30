from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from rhgp.models.baselines import always_a_proba, persistence_proba
from rhgp.models.train import feature_columns, time_split


def evaluate_threshold(
    y_true: np.ndarray, p_fail: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (p_fail >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    return {
        "threshold": threshold,
        "precision_fail": float(p),
        "recall_fail": float(r),
        "f1_fail": float(f1),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate model and baselines.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--test-start", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.data)
    _, test_df = time_split(df, test_start=args.test_start)
    X_test = test_df[feature_columns()]
    y_test = test_df["y_t1"].astype(int).to_numpy()

    model = joblib.load(args.model)
    p_fail = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, object] = {"n_test": int(len(test_df))}
    metrics["logreg"] = evaluate_threshold(y_test, p_fail, threshold=args.threshold)

    p_fail_always_a = always_a_proba(test_df).to_numpy()
    metrics["always_a"] = evaluate_threshold(y_test, p_fail_always_a, threshold=args.threshold)

    p_fail_persist = persistence_proba(test_df).to_numpy()
    metrics["persistence"] = evaluate_threshold(y_test, p_fail_persist, threshold=args.threshold)

    y_pred = (p_fail >= args.threshold).astype(int)
    metrics["logreg_report"] = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division="0",
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
