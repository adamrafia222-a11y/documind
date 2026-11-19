"""STEP 2 (OCR) - fine-tune a text-line recognizer (TrOCR) on the crops of the three datasets  ->  models/ocr_trocr

    python -m documind.data.prepare_ocr                                      # STEP 1: crops + labels (once)
    python -m documind.training.train_ocr --smoke                            # 1 minute: checks that everything runs (do this first)
    python -m documind.training.train_ocr                                    # the real run (microsoft/trocr-small-printed, 4 epochs)
    python -m documind.training.train_ocr --base microsoft/trocr-base-handwritten --batch 8     # bigger / better for handwriting (cheques)
    python -m documind.training.train_ocr --sources cheques --epochs 6       # a model for one source only
    python -m documind.training.evaluate_ocr                                 # STEP 3: Tesseract vs the new model, on crops it never saw

What it does: starts from a model that already reads text (downloaded once from Hugging Face, about 250 MB for the small one), shows it your crops
(with light augmentation: brightness, blur, small rotation, lower resolution) and corrects it until it writes the right transcription. After
every epoch the model is scored on validation crops (CER = share of characters wrong); the best one is saved in models/ocr_trocr, with its
log in reports/ocr_train_log.json. Training stops early when the score has not improved for --patience epochs.

Needs: pip install -r requirements.txt (transformers, sentencepiece, accelerate) next to your GPU build of PyTorch, internet access to
huggingface.co on the first run. Mixed precision (bfloat16) is used on GPUs that support it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from PIL import Image, ImageFilter

from ..config import MODELS_DIR, REPORTS_DIR
from ..data.prepare_ocr import DEFAULT_OUT
from ..ocr.metrics import summarize

MODEL_OUT = MODELS_DIR / "ocr_trocr"
LOG_PATH = REPORTS_DIR / "ocr_train_log.json"


class CropDataset:
    def __init__(self, rows, processor, max_len: int, train: bool):
        self.rows = rows
        self.processor = processor
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch

        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.train:
            img = augment(img, random.Random())
        pixel = self.processor(images=img, return_tensors="pt").pixel_values[0]
        ids = self.processor.tokenizer(r["text"], padding="max_length", max_length=self.max_len, truncation=True).input_ids
        labels = torch.tensor([t if t != self.processor.tokenizer.pad_token_id else -100 for t in ids])
        return {"pixel_values": pixel, "labels": labels}


def read_labels(root: Path, split: str, sources: list[str] | None = None) -> list[dict]:
    """labels.tsv of a split -> [{"path", "text", "source", "kind"}] (only the wanted sources)."""
    f = Path(root) / split / "labels.tsv"
    rows = []
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 4 and (not sources or parts[2] in sources):
                rows.append({"path": Path(root) / split / parts[0], "text": parts[1], "source": parts[2], "kind": parts[3]})
    return rows


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Light, realistic damage so the model does not learn the generators' perfect pixels: lower resolution, blur, small rotation, contrast, margins."""
    from PIL import ImageEnhance, ImageOps
    if rng.random() < 0.35:                                          # lower resolution, as a small phone picture
        k = rng.uniform(0.45, 0.9)
        img = img.resize((max(8, int(img.width * k)), max(8, int(img.height * k))), Image.BILINEAR).resize(img.size, Image.BILINEAR)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BILINEAR, expand=True, fillcolor=img.getpixel((0, 0)))
    if rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.25:
        img = ImageOps.expand(img, border=rng.randint(2, 10), fill=img.getpixel((0, 0)))
    return img


def build_dataset(rows, processor, max_len: int, train: bool):
    return CropDataset(rows, processor, max_len, train)


def evaluate_model(model, processor, rows, device, batch: int = 32, num_beams: int = 1, max_new_tokens: int = 72) -> dict:
    """Read the crops with the model -> {"all": summary, "by_source": {source: summary}} (CER / WER / exact)."""
    import torch
    model.eval()
    hyps: list[str] = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            imgs = [Image.open(r["path"]).convert("RGB") for r in rows[i:i + batch]]
            pixel = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                ids = model.generate(pixel, num_beams=num_beams, max_new_tokens=max_new_tokens)
            hyps += [" ".join(t.split()) for t in processor.batch_decode(ids, skip_special_tokens=True)]
    pairs = [(r["text"], h) for r, h in zip(rows, hyps)]
    by_source = {s: summarize([p for p, r in zip(pairs, rows) if r["source"] == s]) for s in sorted({r["source"] for r in rows})}
    return {"all": summarize(pairs), "by_source": by_source}


def load_model(base: str):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base)
    tok, cfg = processor.tokenizer, model.config
    # keep the values of the pretrained model (it was trained with them); only fill in what is missing
    if cfg.decoder_start_token_id is None:
        cfg.decoder_start_token_id = tok.cls_token_id
    if cfg.pad_token_id is None:
        cfg.pad_token_id = tok.pad_token_id
    if cfg.eos_token_id is None:
        cfg.eos_token_id = tok.sep_token_id
    cfg.vocab_size = cfg.decoder.vocab_size
    return processor, model


def train(root: Path = DEFAULT_OUT, out: Path = MODEL_OUT, base: str = "microsoft/trocr-small-printed", sources: list[str] | None = None, epochs: int = 4,
          batch: int = 16, lr: float = 4e-5, patience: int = 2, eval_samples: int = 1500, max_len: int = 72, workers: int = 0, seed: int = 0,
          smoke: bool = False, log_path: Path = LOG_PATH, device: str | None = None) -> dict:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = read_labels(root, "train", sources), read_labels(root, "val", sources)
    if not tr or not va:
        raise SystemExit(f"No crops in {root} (train {len(tr)}, val {len(va)}). Run first:  python -m documind.data.prepare_ocr")
    random.Random(seed).shuffle(tr)
    random.Random(seed).shuffle(va)
    if smoke:
        tr, va, epochs, batch, workers = tr[:64], va[:32], 1, 4, 0
    va = va[:eval_samples]
    print(f"device {device} | train {len(tr)} crops, val {len(va)} | base {base} | {epochs} epoch(s), batch {batch}, lr {lr}")
    processor, model = load_model(base)
    model.to(device)
    dl = torch.utils.data.DataLoader(build_dataset(tr, processor, max_len, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=len(tr) > batch,
                                     pin_memory=device == "cuda", persistent_workers=workers > 0)
    steps = max(1, epochs * len(dl))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.02, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))))
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    history, best, bad, step, t0 = [], float("inf"), 0, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for i, b in enumerate(dl, 1):
            pixel, labels = b["pixel_values"].to(device), b["labels"].to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(pixel_values=pixel, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run, step = run + float(loss), step + 1
            if i % 100 == 0 or i == len(dl):
                done = step / steps
                print(f"epoch {ep}/{epochs}  step {i}/{len(dl)}  loss {run / i:.4f}  elapsed {(time.time() - t0) / 60:.1f} min  ETA {((time.time() - t0) / max(done, 1e-9) * (1 - done)) / 60:.0f} min", flush=True)
        ev = evaluate_model(model, processor, va, device)
        cer_now = ev["all"]["cer"]
        history.append({"epoch": ep, "train_loss": round(run / max(1, len(dl)), 4), "val": ev})
        print(f"==> epoch {ep}: val CER {cer_now:.4f}  exact {ev['all']['exact']:.3f}   " + "  ".join(f"{s} {v['cer']:.4f}" for s, v in ev["by_source"].items()), flush=True)
        if cer_now < best:
            best, bad = cer_now, 0
            Path(out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            processor.save_pretrained(out)
            print(f"    best so far: saved in {out}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"    no progress for {patience} epoch(s): stop")
                break
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps({"base": base, "sources": sources, "history": history}, indent=2), encoding="utf-8")
    Path(log_path).write_text(json.dumps({"base": base, "sources": sources, "history": history, "best_val_cer": best}, indent=2), encoding="utf-8")
    if smoke:
        print("SMOKE OK: data, model, training step, evaluation and saving all work. Now run without --smoke.")
    print(f"\nBest validation CER {best:.4f} -> {out}\nNext: python -m documind.training.evaluate_ocr")
    return {"best_cer": best, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_OUT), help="crops folder made by prepare_ocr")
    ap.add_argument("--out", default=str(MODEL_OUT))
    ap.add_argument("--base", default="microsoft/trocr-small-printed", help="pretrained model: microsoft/trocr-small-printed (fast) / microsoft/trocr-base-printed / microsoft/trocr-base-handwritten")
    ap.add_argument("--sources", default="", help="comma separated subset of receipts, cheques, fatura (default: all that were prepared)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="lower on 'CUDA out of memory' (8 for the base models)")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--patience", type=int, default=2, help="epochs without improvement before stopping")
    ap.add_argument("--eval-samples", type=int, default=1500, help="validation crops read after every epoch")
    ap.add_argument("--workers", type=int, default=0, help="data-loader processes (use 0 on Windows by default; set >0 only when you want multiprocessing)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: automatic)")
    ap.add_argument("--smoke", action="store_true", help="tiny run (64 crops, 1 epoch) to check that everything works")
    a = ap.parse_args()
    train(Path(a.data), Path(a.out), a.base, [s for s in a.sources.split(",") if s] or None, a.epochs, a.batch, a.lr, a.patience, a.eval_samples, workers=a.workers,
          smoke=a.smoke, device=a.device)


if __name__ == "__main__":
    main()
