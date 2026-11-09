"""FastAPI backend.   Run:  uvicorn documind.api.main:app --reload --port 8000   (docs: http://localhost:8000/docs)"""
from __future__ import annotations

import io
import time
import zipfile

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import MAX_UPLOAD_MB, MODEL_PATH
from ..inference.extractor import ModelNotFound
from ..inference.service import CSV_NAMES, DocumentService
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

_service = DocumentService()
DOC_TYPE = Query("auto", pattern="^(auto|invoice|receipt)$", description="auto = the router decides; or force invoice / receipt")


async def read_upload(f: UploadFile) -> bytes:
    data = await f.read()
    if not data:
        raise HTTPException(400, f"{f.filename}: empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB} MB")
    return data


def _run(data: bytes, filename: str, doc_type: str) -> dict:
    try:
        return _service.extract(data, filename, doc_type)
    except ModelNotFound as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    av = _service.available()
    return {"status": "ok", "model_present": any(av.values()), "models": av, "model_loaded": _service.invoice_loaded}


@app.get("/model")
def model_info():
    """Invoice model (classes, columns, meta) at the top level as before, and the CSV columns of every type under `types`."""
    av = _service.available()
    out: dict = {"models": av, "types": {}}
    if av["invoice"]:
        ex = _run_invoice_model()
        out.update({"classes": ex.classes, "columns": ex.columns, "meta": ex.meta})
        out["types"]["invoice"] = {"columns": ex.columns, "csv": CSV_NAMES["invoice"]}
    if av["receipt"]:
        out["types"]["receipt"] = {"columns": _service.columns("receipt"), "csv": CSV_NAMES["receipt"]}
    if not any(av.values()):
        raise HTTPException(503, "No model is installed yet.")
    return out


def _run_invoice_model():
    try:
        return _service.invoice()
    except ModelNotFound as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/extract")
async def extract(file: UploadFile = File(...), format: str = Query("json", pattern="^(json|csv)$"), doc_type: str = DOC_TYPE,
                  user=Depends(accounts.user_or_none)):
    """One image or PDF -> JSON (fields + row + doc_type) or a one-line CSV that only has the columns of its own type."""
    data = await read_upload(file)
    t0 = time.perf_counter()
    try:
        result = _run(data, file.filename or "document", doc_type)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    accounts.save_history(user, result, int((time.perf_counter() - t0) * 1000), data)
    if format == "csv":
        return Response(_service.to_csv(result["doc_type"], [result]), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{CSV_NAMES[result["doc_type"]]}"'})
    return result


@app.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...), format: str = Query("csv", pattern="^(json|csv)$"), doc_type: str = DOC_TYPE):
    """Many files -> ONE CSV PER DOCUMENT TYPE (never mixed).
    format=csv : only one type in the batch -> invoices.csv or receipts.csv;  several types -> extractions.zip with one csv per type.
    format=json: {"results": [...], "errors": [...], "csv_files": {"invoices.csv": n_rows, ...}}."""
    results, errors = [], []
    for f in files:
        try:
            results.append(_run(await read_upload(f), f.filename or "document", doc_type))
        except (ValueError, HTTPException) as exc:
            errors.append({"file": f.filename, "error": getattr(exc, "detail", str(exc))})
    counts: dict[str, int] = {}
    for r in results:
        counts[CSV_NAMES[r["doc_type"]]] = counts.get(CSV_NAMES[r["doc_type"]], 0) + 1
    if format == "json":
        return {"results": results, "errors": errors, "csv_files": counts}
    csvs = _service.csv_files(results) if results else {}
    headers = {"X-Errors": str(len(errors)), "X-Doc-Types": ",".join(sorted({r["doc_type"] for r in results}))}
    if len(csvs) == 1:
        name, content = next(iter(csvs.items()))
        return Response(content, media_type="text/csv; charset=utf-8", headers={**headers, "Content-Disposition": f'attachment; filename="{name}"'})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in csvs.items():
            z.writestr(name, content)
    return Response(buf.getvalue(), media_type="application/zip", headers={**headers, "Content-Disposition": 'attachment; filename="extractions.zip"'})


# ---- web interface (folder `src/` at the project root): http://localhost:8000/app/ ----
_FRONTEND = Path(__file__).resolve().parents[2] / "src"
if _FRONTEND.is_dir():
    app.mount("/app", _NoCacheStatic(directory=_FRONTEND, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")
