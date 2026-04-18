"""AWS Lambda handler for Mbay translation inference.

Loads a fine-tuned NLLB-600M model (optionally INT8-quantized) and serves
translation requests via FastAPI + Mangum.

Expected environment:
    MODEL_DIR: path to model directory (default: /opt/ml/model)

Deployment: Docker image on AWS Lambda (10GB, 10GB RAM).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))

SUPPORTED_PAIRS = {
    ("eng", "mbay"),
    ("fra", "mbay"),
    ("mbay", "eng"),
    ("mbay", "fra"),
}

# Maps short codes to NLLB lang codes.
LANG_TO_NLLB = {
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "mbay": "myb_Latn",
}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"Loading model from {MODEL_DIR}")
    t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    )
    _model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Mbay Translator", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source_lang: str = Field(..., pattern=r"^(eng|fra|mbay)$")
    target_lang: str = Field(..., pattern=r"^(eng|fra|mbay)$")


class TranslateResponse(BaseModel):
    translation: str
    source_lang: str
    target_lang: str
    latency_ms: int


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    pair = (req.source_lang, req.target_lang)
    if pair not in SUPPORTED_PAIRS:
        raise HTTPException(400, f"Unsupported pair: {pair}")

    _load_model()
    t0 = time.time()

    src_nllb = LANG_TO_NLLB[req.source_lang]
    tgt_nllb = LANG_TO_NLLB[req.target_lang]

    _tokenizer.src_lang = src_nllb
    inputs = _tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)

    tgt_token_id = _tokenizer.convert_tokens_to_ids(tgt_nllb)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_new_tokens=256,
            num_beams=4,
            length_penalty=1.0,
        )

    translation = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    latency_ms = int((time.time() - t0) * 1000)

    return TranslateResponse(
        translation=translation,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


# AWS Lambda entry point
lambda_handler = Mangum(app, lifespan="on")
