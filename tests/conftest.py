"""Tests never download a pretrained OCR model (~1.3 GB) and never depend on one being installed."""
import pytest


@pytest.fixture(autouse=True)
def _no_pretrained_ocr_download(monkeypatch):
    monkeypatch.setenv("DOCUMIND_PRETRAINED_OCR", "off")
