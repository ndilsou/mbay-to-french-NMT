"""Fine-tune a fresh MMS CTC adapter for toned Mbay ASR.

The dictionary data is stored as NFC text, but the ASR tokenizer is built from
NFD-normalized transcripts. That keeps base letters and tone/diacritic marks as
separate CTC tokens, avoiding a large precomposed-character vocabulary while
still round-tripping the toned orthography after NFC normalization.
"""

from __future__ import annotations

import argparse
import json
import os

# aten::_ctc_loss is not implemented on MPS; fall back to CPU for that op.
# Must be set before torch dispatches any MPS op.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import random
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    set_seed,
)

TARGET_LANG = "myb_dict"
SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "facebook/mms-1b-all"
DEFAULT_KINDS = ("sentence", "expression", "headword")
WORD_DELIMITER = "|"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


@dataclass(frozen=True)
class AsrExample:
    text: str
    audio_path: Path
    s3_key: str
    kind: str


def normalize_text_for_ctc(text: str) -> str:
    """Use NFD so combining tone marks become independent CTC labels."""
    return unicodedata.normalize("NFD", str(text))


def build_vocab_dict(texts: Iterable[str]) -> dict[str, int]:
    chars: set[str] = set()
    for text in texts:
        chars.update(normalize_text_for_ctc(text))

    chars.discard(" ")
    ordered = sorted(chars)
    vocab = {char: idx for idx, char in enumerate(ordered)}
    vocab[WORD_DELIMITER] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    vocab[PAD_TOKEN] = len(vocab)
    return vocab


def make_processor(vocab: dict[str, int], output_dir: Path | None = None) -> Wav2Vec2Processor:
    if output_dir is None:
        tmp = tempfile.TemporaryDirectory()
        vocab_dir = Path(tmp.name)
    else:
        tmp = None
        vocab_dir = output_dir
        vocab_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = vocab_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        word_delimiter_token=WORD_DELIMITER,
        bos_token=None,
        eos_token=None,
    )
    tokenizer._tmp_vocab_dir = tmp  # Keep tempdir alive when output_dir is omitted.
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to decode the Mbay MP3 audio files. "
            "Install ffmpeg and ensure it is on PATH before training."
        )


def decode_audio_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a 16kHz mono PCM wav without ffmpeg (pre-transcoded corpus)."""
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != sample_rate or w.getnchannels() != 1:
            raise RuntimeError(
                f"{path}: expected {sample_rate}Hz mono wav, got "
                f"{w.getframerate()}Hz {w.getnchannels()}ch"
            )
        if w.getsampwidth() != 2:
            raise RuntimeError(f"{path}: expected 16-bit PCM wav")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def decode_audio_ffmpeg(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        return decode_audio_wav(path, sample_rate)
    check_ffmpeg()
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {detail}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio samples from {path}")
    return audio


def audio_duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        import wave

        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()

    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    return float(decode_audio_ffmpeg(path).shape[0]) / SAMPLE_RATE


def parse_kinds(raw: str) -> set[str]:
    kinds = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = kinds - set(DEFAULT_KINDS)
    if unknown:
        raise ValueError(f"Unknown kind(s): {', '.join(sorted(unknown))}")
    return kinds


def load_examples(
    manifest: Path,
    audio_dir: Path,
    kinds: set[str],
    min_audio_seconds: float,
    max_audio_seconds: float,
) -> list[AsrExample]:
    df = pd.read_parquet(manifest)
    required = {"language", "text", "kind", "s3_key"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing_cols))}")

    df = df[(df["language"] == "mbay") & df["s3_key"].notna() & df["kind"].isin(kinds)].copy()
    if df.empty:
        raise ValueError("No mbay rows with non-null s3_key matched the requested kinds")

    missing_audio: list[Path] = []
    examples: list[AsrExample] = []
    skipped_duration = 0
    for row in df.itertuples(index=False):
        s3_key = str(row.s3_key)
        audio_path = audio_dir / Path(s3_key).name
        wav_path = audio_path.with_suffix(".wav")
        if wav_path.exists():
            audio_path = wav_path
        elif not audio_path.exists():
            missing_audio.append(audio_path)
            continue

        duration = audio_duration_seconds(audio_path)
        if duration < min_audio_seconds or duration > max_audio_seconds:
            skipped_duration += 1
            continue

        examples.append(
            AsrExample(
                text=str(row.text),
                audio_path=audio_path,
                s3_key=s3_key,
                kind=str(row.kind),
            )
        )

    if missing_audio:
        # ~10% of scraped files are HTML error pages that never transcoded;
        # skip them, but refuse to train if the corpus is mostly absent.
        frac = len(missing_audio) / max(len(missing_audio) + len(examples) + skipped_duration, 1)
        preview = "\n".join(str(path) for path in missing_audio[:5])
        if frac > 0.2:
            raise FileNotFoundError(
                f"{len(missing_audio)} audio files ({frac:.0%}) referenced by the manifest were not "
                f"found under {audio_dir}. First missing paths:\n{preview}"
            )
        print(f"WARNING: skipping {len(missing_audio)} manifest rows with missing audio ({frac:.1%})")
    if not examples:
        raise ValueError(
            "All matching rows were filtered out by duration. "
            f"Skipped {skipped_duration} rows outside [{min_audio_seconds}, {max_audio_seconds}] seconds."
        )
    if skipped_duration:
        print(f"Skipped {skipped_duration} rows outside duration bounds")
    return examples


def split_by_text(
    examples: list[AsrExample],
    eval_fraction: float,
    seed: int,
) -> tuple[list[AsrExample], list[AsrExample]]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be greater than 0 and less than 1")

    groups: dict[str, list[AsrExample]] = {}
    for example in examples:
        groups.setdefault(normalize_text_for_ctc(example.text), []).append(example)

    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)

    target_eval_rows = max(1, round(len(examples) * eval_fraction))
    eval_examples: list[AsrExample] = []
    train_examples: list[AsrExample] = []
    for group in grouped:
        if len(eval_examples) < target_eval_rows:
            eval_examples.extend(group)
        else:
            train_examples.extend(group)

    if not train_examples:
        raise ValueError("Split produced no training rows; lower --eval-fraction or provide more data")
    if not eval_examples:
        raise ValueError("Split produced no eval rows; increase --eval-fraction or provide more data")
    return train_examples, eval_examples


class MbayAsrDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[AsrExample], processor: Wav2Vec2Processor):
        self.examples = examples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int] | np.ndarray]:
        example = self.examples[idx]
        audio = decode_audio_ffmpeg(example.audio_path)
        inputs = self.processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE)
        labels = self.processor.tokenizer(normalize_text_for_ctc(example.text)).input_ids
        return {"input_values": inputs["input_values"][0], "labels": labels}


class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, features: list[dict[str, list[int] | np.ndarray]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        cur = [i]
        for j, item_b in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (item_a != item_b),
                )
            )
        prev = cur
    return prev[-1]


def _error_rate(predictions: list[str], references: list[str], unit: str) -> float:
    edits = 0
    total = 0
    for pred, ref in zip(predictions, references, strict=True):
        if unit == "word":
            pred_units = pred.split()
            ref_units = ref.split()
        else:
            pred_units = list(pred)
            ref_units = list(ref)
        edits += _edit_distance(pred_units, ref_units)
        total += len(ref_units)
    return edits / total if total else 0.0


def make_compute_metrics(processor: Wav2Vec2Processor):
    def compute_metrics(eval_pred) -> dict[str, float]:
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.asarray(pred_ids)

        label_ids = np.asarray(eval_pred.label_ids).copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        pred_str = [unicodedata.normalize("NFC", text).strip() for text in pred_str]
        label_str = [unicodedata.normalize("NFC", text).strip() for text in label_str]
        return {
            "cer": _error_rate(pred_str, label_str, "char"),
            "wer": _error_rate(pred_str, label_str, "word"),
        }

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def freeze_for_mms_adapter_training(
    model: Wav2Vec2ForCTC, scope: str = "adapter", encoder_top_layers: int = 12
) -> tuple[int, int]:
    """Mark parameters trainable per scope.

    adapter: MMS adapter layers + lm_head only (~2.5M params).
    encoder-top: adapter + lm_head + top N transformer layers.
    full: everything except the convolutional feature extractor.
    """
    for param in model.parameters():
        param.requires_grad = False

    num_layers = model.config.num_hidden_layers
    top_range = {f"wav2vec2.encoder.layers.{i}." for i in range(num_layers - encoder_top_layers, num_layers)}

    def in_scope(name: str) -> bool:
        if "adapter_layer" in name or name.startswith("lm_head."):
            return True
        if scope == "full":
            return not name.startswith("wav2vec2.feature_extractor")
        if scope == "encoder-top":
            return any(name.startswith(prefix) for prefix in top_range)
        return False

    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if in_scope(name):
            param.requires_grad = True
            trainable += param.numel()

    if trainable == 0:
        raise RuntimeError("No parameters were marked trainable")
    return trainable, total


def save_mms_adapter(model: Wav2Vec2ForCTC, output_dir: Path, target_lang: str = TARGET_LANG) -> Path:
    adapter_state = {name: tensor.detach().cpu() for name, tensor in model._get_adapters().items()}
    safe_path = output_dir / f"adapter.{target_lang}.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(adapter_state, str(safe_path))
        return safe_path
    except Exception as exc:
        bin_path = output_dir / f"adapter.{target_lang}.bin"
        torch.save(adapter_state, bin_path)
        print(f"Could not save safetensors adapter ({exc}); wrote {bin_path}")
        return bin_path


def make_model(args: argparse.Namespace, vocab_size: int, pad_token_id: int) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
    )
    model.target_lang = args.target_lang
    model.config.target_lang = args.target_lang
    model.config.pad_token_id = pad_token_id
    model.config.bos_token_id = None
    model.config.eos_token_id = None
    model.config.ctc_zero_infinity = True
    model.init_adapter_layers()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable, total = freeze_for_mms_adapter_training(
        model, scope=args.train_scope, encoder_top_layers=args.encoder_top_layers
    )
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-lang", default=TARGET_LANG)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--warmup-ratio", default=0.1, type=float)
    parser.add_argument("--eval-fraction", default=0.05, type=float)
    parser.add_argument("--max-audio-seconds", default=20.0, type=float)
    parser.add_argument("--min-audio-seconds", default=0.3, type=float)
    parser.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    parser.add_argument("--seed", default=93100, type=int)
    parser.add_argument("--logging-steps", default=25, type=int)
    parser.add_argument("--save-steps", default=500, type=int)
    parser.add_argument("--eval-steps", default=500, type=int)
    parser.add_argument("--dataloader-num-workers", default=0, type=int)
    parser.add_argument(
        "--train-scope",
        default="adapter",
        choices=["adapter", "encoder-top", "full"],
        help="Which parameters to train (feature extractor always frozen)",
    )
    parser.add_argument("--encoder-top-layers", default=12, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    # ffmpeg is only needed for mp3 input; a pre-transcoded 16kHz wav corpus
    # (the SageMaker path) is decoded with the stdlib wave module.
    if not any(args.audio_dir.glob("*.wav")):
        check_ffmpeg()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kinds = parse_kinds(args.kinds)
    examples = load_examples(
        manifest=args.manifest,
        audio_dir=args.audio_dir,
        kinds=kinds,
        min_audio_seconds=args.min_audio_seconds,
        max_audio_seconds=args.max_audio_seconds,
    )
    train_examples, eval_examples = split_by_text(examples, args.eval_fraction, args.seed)
    print(f"Rows: train={len(train_examples)} eval={len(eval_examples)} total={len(examples)}")

    vocab = build_vocab_dict(example.text for example in examples)
    vocab_dir = args.output_dir / "tokenizer-vocab"
    processor = make_processor(vocab, vocab_dir)
    print(f"Built NFD character vocabulary with {len(vocab)} tokens for target_lang={args.target_lang!r}")

    model = make_model(args, vocab_size=len(vocab), pad_token_id=processor.tokenizer.pad_token_id)
    train_dataset = MbayAsrDataset(train_examples, processor)
    eval_dataset = MbayAsrDataset(eval_examples, processor)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        group_by_length=False,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=os.environ.get("TRAINING_JOB_NAME", "mbay-mms-asr-adapter"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        processing_class=processor,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(final_dir)
    adapter_path = save_mms_adapter(trainer.model, final_dir, args.target_lang)
    print(f"Saved final model, processor, and MMS adapter to {final_dir}")
    print(f"Adapter file: {adapter_path}")


if __name__ == "__main__":
    main()
