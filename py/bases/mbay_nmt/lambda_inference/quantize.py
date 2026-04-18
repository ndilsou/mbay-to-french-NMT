"""Offline INT8 quantization for Lambda deployment.

Loads the fine-tuned NLLB model and quantizes it to INT8 using
optimum/onnxruntime or torch dynamic quantization, then saves
the result for baking into the Lambda container.

Usage:
    uv run python -m mbay_nmt.lambda_inference.quantize \\
        --input models/nllb-600m-mbay-finetuned \\
        --output models/nllb-600m-mbay-int8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def quantize_dynamic(input_dir: Path, output_dir: Path) -> None:
    """Apply PyTorch dynamic INT8 quantization (CPU inference)."""
    print(f"Loading model from {input_dir}")
    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(input_dir, torch_dtype=torch.float32)

    print("Applying dynamic INT8 quantization...")
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    quantized.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    orig_size = sum(f.stat().st_size for f in input_dir.rglob("*.safetensors"))
    quant_size = sum(f.stat().st_size for f in output_dir.rglob("*.bin"))
    if not quant_size:
        quant_size = sum(f.stat().st_size for f in output_dir.rglob("*.safetensors"))
    print(f"Original: {orig_size / 1e9:.2f} GB → Quantized: {quant_size / 1e9:.2f} GB")
    print(f"Saved to {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    quantize_dynamic(args.input, args.output)


if __name__ == "__main__":
    main()
