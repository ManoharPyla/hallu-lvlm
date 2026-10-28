"""Runs the offline synthetic smoke pipeline end to end in a temp dir."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_pipeline(tmp_path):
    mod = _load()
    out = tmp_path / "smoke"
    metrics = mod.main(["--out", str(out), "--n-images", "80"])
    assert (out / "metrics.json").exists()
    assert (out / "corrections.jsonl").exists()
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["banner"] == "SYNTHETIC SMOKE TEST — NOT A BENCHMARK RESULT"
    assert saved["synthetic"] is True
    assert "NOT A BENCHMARK RESULT" in (out / "report.md").read_text(encoding="utf-8")
    det = metrics["tables"]["detection"]
    assert {"mlp+identity", "logreg+isotonic"} <= set(det)
    # features carry signal, so the detector must beat chance on synthetic data
    assert det["mlp+platt"]["auroc"] > 0.7
    cor = metrics["tables"]["correction"]
    assert cor["none"]["halluc_removed"] == 0.0  # doing nothing is never rewarded
    assert cor["selective"]["halluc_removed"] > 0.0
    assert 0.0 <= metrics["risk_control"]["threshold"] <= 1.0
