"""FastAPI backend.   Run:  uvicorn documind.api.main:app --reload --port 8000   (docs: http://localhost:8000/docs)"""
from __future__ import annotations

import time

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import MAX_UPLOAD_MB, MODEL_PATH
from ..inference.extractor import FieldExtractor, ModelNotFound
from . import accounts


class _NoCacheStatic(StaticFiles):
    """Static files that the browser must revalidate on every load, so frontend updates show up without a hard refresh."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

app = FastAPI(title="DocuMind AI - extraction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(accounts.router)

_extractor: FieldExtractor | None = None


def get_extractor() -> FieldExtractor:
    global _extractor
    if _extractor is None:
        try:
            _extractor = FieldExtractor()
        except ModelNotFound as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _extractor


async def read_upload(f: UploadFile) -> bytes:
    data = await f.read()
    if not data:
        raise HTTPException(400, f"{f.filename}: empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB} MB")
    return data


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists(), "model_loaded": _extractor is not None}


@app.get("/model")
def model_info():
    ex = get_extractor()
    return {"classes": ex.classes, "columns": ex.columns, "meta": ex.meta}


@app.post("/extract")
async def extract(file: UploadFile = File(...), format: str = Query("json", pattern="^(json|csv)$"),
                  user=Depends(accounts.user_or_none)):
    """One image or PDF -> JSON (fields + row) or a one-line CSV."""
    ex = get_extractor()
    data = await read_upload(file)
    t0 = time.perf_counter()
    try:
        result = ex.extract_bytes(data, file.filename or "document")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    accounts.save_history(user, result, int((time.perf_counter() - t0) * 1000), data)
    if format == "csv":
        return Response(ex.to_csv([result]), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="extraction.csv"'})
    return result


@app.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...), format: str = Query("csv", pattern="^(json|csv)$")):
    """Many files -> one CSV with one row per document (the 'pictures to CSV' endpoint)."""
    ex = get_extractor()
    results, errors = [], []
    for f in files:
        try:
            results.append(ex.extract_bytes(await read_upload(f), f.filename or "document"))
        except (ValueError, HTTPException) as exc:
            errors.append({"file": f.filename, "error": getattr(exc, "detail", str(exc))})
    if format == "json":
        return {"results": results, "errors": errors}
    return Response(ex.to_csv(results), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="extractions.csv"',
                             "X-Errors": str(len(errors))})


# ---- web interface (folder `src/` at the project root): http://localhost:8000/app/ ----
_FRONTEND = Path(__file__).resolve().parents[2] / "src"
if _FRONTEND.is_dir():
    app.mount("/app", _NoCacheStatic(directory=_FRONTEND, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")
