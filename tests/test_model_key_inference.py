from pathlib import Path

from rhgp.models.eval import parse_thresholds


def infer_model_key(model_path: Path) -> str:
    name = model_path.stem.lower()
    if "logreg" in name:
        return "logreg"
    if "rf" in name or "randomforest" in name:
        return "rf"
    return "model"


def test_thresholds_parsing_kept() -> None:
    assert parse_thresholds("0.5,0.7") == [0.5, 0.7]


def test_model_key_inference_logreg_rf() -> None:
    assert infer_model_key(Path("models/logreg.joblib")) == "logreg"
    assert infer_model_key(Path("models/rf.joblib")) == "rf"

