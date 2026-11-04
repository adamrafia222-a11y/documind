"""End-to-end test on the toy dataset (needs the Tesseract program):  pytest -q tests/test_pipeline_toy.py
Uses a temporary folder, so your real data/, models/ and reports/ are not touched."""
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")


def test_prepare_train_extract(tmp_path):
    env = {**os.environ,
           "DOCUMIND_DATA_DIR": str(tmp_path / "data"),
           "DOCUMIND_MODELS_DIR": str(tmp_path / "models"),
           "DOCUMIND_MODEL_PATH": str(tmp_path / "models" / "field_extractor.joblib"),
           "DOCUMIND_REPORTS_DIR": str(tmp_path / "reports")}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, check=True)

    toy = tmp_path / "toy"
    run("scripts/make_toy_data.py", "--out", str(toy), "--per-template", "8")
    run("-m", "documind.data.prepare", "--data", str(toy), "--per-template", "8", "--workers", "2")
    run("-m", "documind.training.train", "--n-estimators", "60")
    assert (tmp_path / "models" / "field_extractor.joblib").exists()
    assert (tmp_path / "reports" / "training_report.json").exists()

    image = next((toy / "images").glob("Template1_*.jpg"))
    out_csv = tmp_path / "out.csv"
    run("-m", "documind.inference.cli", str(image), "-o", str(out_csv))
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8-sig")))
    assert len(rows) == 1 and rows[0]["file"] == image.name
    assert rows[0]["TOTAL"].startswith("$")
