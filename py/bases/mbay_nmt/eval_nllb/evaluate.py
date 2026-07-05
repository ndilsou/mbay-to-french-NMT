"""Evaluate an NLLB Mbay model with forced target-language generation.

Usage:
    uv run python -m mbay_nmt.eval_nllb.evaluate \
        --model-dir models/nllb-600m-mbay-finetuned/final \
        --data datasets/processed/mbay-corpus-v3.parquet \
        --split test \
        --beams 5 \
        --out-dir reports/eval
"""

from __future__ import annotations

import argparse
import json
import os
import random
import unicodedata
from pathlib import Path

os.environ.setdefault("HF_EVALUATE_CACHE", "/tmp/hf-evaluate-cache")
os.environ.setdefault("HF_METRICS_CACHE", "/tmp/hf-metrics-cache")

import evaluate
import pandas as pd
import torch
from sacrebleu.metrics import CHRF as SacrebleuCHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mbay_nmt.tokenizer.nllb_extend import PIPELINE_TO_NLLB

CHRF = evaluate.load("chrf")
BLEU = evaluate.load("sacrebleu")
SACREBLEU_CHRF = SacrebleuCHRF(word_order=2)
MBAY_LANGS = {"mbay", "eng", "fra"}


def nfc_text(text: object) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def strip_combining_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def chrf_score(predictions: list[str], references: list[str], *, strip_marks: bool = False) -> float:
    predictions = [nfc_text(prediction) for prediction in predictions]
    references = [nfc_text(reference) for reference in references]
    if strip_marks:
        predictions = [strip_combining_marks(prediction) for prediction in predictions]
        references = [strip_combining_marks(reference) for reference in references]
    return float(SACREBLEU_CHRF.corpus_score(predictions, [references]).score)


def metric_bucket(source_lang: str, target_lang: str) -> str:
    if target_lang == "fra" and source_lang not in MBAY_LANGS:
        return "aux->fra"
    return f"{source_lang}->{target_lang}"


def load_model(model_dir: str | Path) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_predictions(
    model_dir: str | Path,
    df: pd.DataFrame,
    *,
    beams: int,
    batch_size: int = 32,
    max_new_tokens: int = 128,
) -> pd.DataFrame:
    tokenizer, model, device = load_model(model_dir)
    source_max_length = tokenizer.model_max_length
    if source_max_length > 4096:
        source_max_length = 1024

    parts = []
    with torch.no_grad():
        for (source_lang, target_lang), direction_df in df.groupby(["source_lang", "target_lang"], sort=True):
            src_nllb = PIPELINE_TO_NLLB.get(source_lang, source_lang)
            tgt_nllb = PIPELINE_TO_NLLB.get(target_lang, target_lang)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
            if forced_bos_token_id == tokenizer.unk_token_id:
                raise ValueError(f"Target language token {tgt_nllb!r} resolves to UNK")

            predictions = []
            for start in range(0, len(direction_df), batch_size):
                batch = direction_df.iloc[start : start + batch_size]
                tokenizer.src_lang = src_nllb
                inputs = tokenizer(
                    batch["source_text"].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=source_max_length,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=beams,
                )
                predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

            out = direction_df.copy()
            out["prediction"] = [prediction.strip() for prediction in predictions]
            out["bucket"] = [metric_bucket(source_lang, target_lang)] * len(out)
            parts.append(out)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compute_metrics(predicted: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for bucket, bucket_df in predicted.groupby("bucket", sort=True):
        predictions = [nfc_text(value) for value in bucket_df["prediction"].tolist()]
        references = [nfc_text(value) for value in bucket_df["target_text"].tolist()]
        chrf = CHRF.compute(predictions=predictions, references=references, word_order=2)["score"]
        bleu = BLEU.compute(predictions=predictions, references=[[ref] for ref in references])["score"]
        values: dict[str, float | int] = {
            "n": int(len(bucket_df)),
            "chrf++": round(float(chrf), 4),
            "bleu": round(float(bleu), 4),
        }
        if (bucket_df["target_lang"] == "mbay").all():
            stripped_chrf = chrf_score(predictions, references, strip_marks=True)
            values["chrf_raw"] = round(float(chrf), 4)
            values["chrf_stripped"] = round(stripped_chrf, 4)
        metrics[bucket] = values
    return metrics


def _aligned_bucket_frames(
    model_predicted: pd.DataFrame,
    baseline_predicted: pd.DataFrame,
    bucket: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_bucket = model_predicted[model_predicted["bucket"] == bucket].copy()
    baseline_bucket = baseline_predicted[baseline_predicted["bucket"] == bucket].copy()
    if "_eval_row_id" in model_bucket.columns and "_eval_row_id" in baseline_bucket.columns:
        model_bucket = model_bucket.sort_values("_eval_row_id", kind="mergesort").reset_index(drop=True)
        baseline_bucket = baseline_bucket.sort_values("_eval_row_id", kind="mergesort").reset_index(drop=True)
    else:
        model_bucket = model_bucket.reset_index(drop=True)
        baseline_bucket = baseline_bucket.reset_index(drop=True)
    if len(model_bucket) != len(baseline_bucket):
        raise ValueError(f"Cannot compare bucket {bucket!r}: model n={len(model_bucket)}, baseline n={len(baseline_bucket)}")
    return model_bucket, baseline_bucket


def paired_bootstrap_chrf_p_value(
    model_predictions: list[str],
    baseline_predictions: list[str],
    references: list[str],
    *,
    strip_marks: bool = False,
    resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    model_predictions = [nfc_text(value) for value in model_predictions]
    baseline_predictions = [nfc_text(value) for value in baseline_predictions]
    references = [nfc_text(value) for value in references]
    if strip_marks:
        model_predictions = [strip_combining_marks(value) for value in model_predictions]
        baseline_predictions = [strip_combining_marks(value) for value in baseline_predictions]
        references = [strip_combining_marks(value) for value in references]

    observed_delta = chrf_score(model_predictions, references) - chrf_score(baseline_predictions, references)
    if not references:
        return observed_delta, 1.0

    rng = random.Random(seed)
    opposite_or_zero = 0
    n = len(references)
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_model = [model_predictions[index] for index in indices]
        sample_baseline = [baseline_predictions[index] for index in indices]
        sample_references = [references[index] for index in indices]
        sample_delta = chrf_score(sample_model, sample_references) - chrf_score(sample_baseline, sample_references)
        if observed_delta >= 0:
            opposite_or_zero += sample_delta <= 0
        else:
            opposite_or_zero += sample_delta >= 0
    p_value = (opposite_or_zero + 1) / (resamples + 1)
    return observed_delta, p_value


def compute_significance(
    model_predicted: pd.DataFrame,
    baseline_predicted: pd.DataFrame,
    *,
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float | bool | int]]:
    significance: dict[str, dict[str, float | bool | int]] = {}
    for bucket in sorted(set(model_predicted["bucket"])):
        model_bucket, baseline_bucket = _aligned_bucket_frames(model_predicted, baseline_predicted, bucket)
        references = [nfc_text(value) for value in model_bucket["target_text"].tolist()]
        model_predictions = [nfc_text(value) for value in model_bucket["prediction"].tolist()]
        baseline_predictions = [nfc_text(value) for value in baseline_bucket["prediction"].tolist()]
        delta, p_value = paired_bootstrap_chrf_p_value(
            model_predictions,
            baseline_predictions,
            references,
            resamples=resamples,
            seed=seed,
        )
        values: dict[str, float | bool | int] = {
            "n": int(len(model_bucket)),
            "chrf_delta": round(delta, 4),
            "chrf_p_value": round(p_value, 6),
            "chrf_significant": p_value < 0.05,
        }
        if (model_bucket["target_lang"] == "mbay").all():
            stripped_delta, stripped_p_value = paired_bootstrap_chrf_p_value(
                model_predictions,
                baseline_predictions,
                references,
                strip_marks=True,
                resamples=resamples,
                seed=seed,
            )
            values["chrf_stripped_delta"] = round(stripped_delta, 4)
            values["chrf_stripped_p_value"] = round(stripped_p_value, 6)
            values["chrf_stripped_significant"] = stripped_p_value < 0.05
        significance[bucket] = values
    return significance


def write_samples(predicted: pd.DataFrame, out_path: Path, seed: int = 42) -> None:
    rows = []
    mbay_df = predicted[(predicted["source_lang"] == "mbay") | (predicted["target_lang"] == "mbay")]
    for _, direction_df in mbay_df.groupby(["source_lang", "target_lang"], sort=True):
        sample = direction_df.sample(n=min(30, len(direction_df)), random_state=seed)
        rows.append(
            sample.rename(
                columns={
                    "source_text": "source",
                    "target_text": "reference",
                }
            )[["source_lang", "target_lang", "source", "reference", "prediction"]]
        )
    samples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["source_lang", "target_lang", "source", "reference", "prediction"]
    )
    samples.to_csv(out_path, index=False)


def write_report(
    out_path: Path,
    *,
    model_dir: str,
    split: str,
    model_metrics: dict[str, dict[str, float | int]],
    baseline_dir: str | None = None,
    baseline_metrics: dict[str, dict[str, float | int]] | None = None,
    significance: dict[str, dict[str, float | bool | int]] | None = None,
) -> None:
    lines = [
        f"# NLLB Evaluation ({split})",
        "",
        f"Model: `{model_dir}`",
    ]
    if baseline_dir:
        lines.append(f"Baseline: `{baseline_dir}`")
    lines.extend(["", "| Direction | n | chrF++ | chrF++ stripped | BLEU |"])
    lines.append("|---|---:|---:|---:|---:|")
    for bucket, values in sorted(model_metrics.items()):
        stripped = values.get("chrf_stripped")
        stripped_text = f"{stripped:.2f}" if isinstance(stripped, float) else ""
        lines.append(
            f"| {bucket} | {values['n']} | {values['chrf++']:.2f} | {stripped_text} | {values['bleu']:.2f} |"
        )

    if baseline_metrics:
        lines.extend(
            [
                "",
                "## Baseline Delta",
                "",
                "* p<0.05, paired bootstrap over sentence pairs (1000 resamples).",
                "",
                "| Direction | n | Metric | baseline | model | delta | p |",
            ]
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for bucket, values in sorted(model_metrics.items()):
            base = baseline_metrics.get(bucket)
            if base is None:
                continue
            sig = significance.get(bucket, {}) if significance else {}
            chrf_star = "*" if sig.get("chrf_significant") else ""
            chrf_p = sig.get("chrf_p_value")
            chrf_p_text = f"{chrf_p:.4f}" if isinstance(chrf_p, (float, int)) else ""
            lines.append(
                f"| {bucket} | {values['n']} | chrF++ | {base['chrf++']:.2f} | {values['chrf++']:.2f} | "
                f"{values['chrf++'] - base['chrf++']:.2f}{chrf_star} | "
                f"{chrf_p_text} |"
            )
            if "chrf_stripped" in values and "chrf_stripped" in base:
                stripped_star = "*" if sig.get("chrf_stripped_significant") else ""
                stripped_p = sig.get("chrf_stripped_p_value")
                stripped_p_text = f"{stripped_p:.4f}" if isinstance(stripped_p, (float, int)) else ""
                lines.append(
                    f"| {bucket} | {values['n']} | chrF++ stripped | {base['chrf_stripped']:.2f} | "
                    f"{values['chrf_stripped']:.2f} | "
                    f"{values['chrf_stripped'] - base['chrf_stripped']:.2f}{stripped_star} | "
                    f"{stripped_p_text} |"
                )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline", default=None, help="Optional second model dir to compare against")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No rows found for split={args.split!r} in {args.data}")
    df["_eval_row_id"] = range(len(df))

    predicted = generate_predictions(args.model_dir, df, beams=args.beams)
    model_metrics = compute_metrics(predicted)
    metrics_payload: dict[str, object] = {"model": model_metrics}

    baseline_metrics = None
    significance = None
    if args.baseline:
        baseline_predicted = generate_predictions(args.baseline, df, beams=args.beams)
        baseline_metrics = compute_metrics(baseline_predicted)
        significance = compute_significance(predicted, baseline_predicted)
        metrics_payload["baseline"] = baseline_metrics
        metrics_payload["significance"] = significance

    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n")
    write_report(
        out_dir / "report.md",
        model_dir=args.model_dir,
        split=args.split,
        model_metrics=model_metrics,
        baseline_dir=args.baseline,
        baseline_metrics=baseline_metrics,
        significance=significance,
    )
    write_samples(predicted, out_dir / "samples.csv")

    print(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    print(f"Wrote evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
