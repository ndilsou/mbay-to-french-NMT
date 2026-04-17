"""Paraphrase augmentation for the Mbay parallel corpus.

For each Mbay→{fra,eng} training row, paraphrase the target side using
Gemini Flash-Lite, producing variant translations that expand training
signal without needing new Mbay text.

CLI usage:
    uv run python -m mbay_nmt.data_sources.augment \
        --input datasets/processed/mbay-corpus-v1.parquet \
        --output datasets/processed/mbay-corpus-v1-aug.parquet \
        --multiplier 3 --sample-validate 200
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH_SIZE = 20
_RPM_LIMIT = 60
_MIN_INTERVAL = 60.0 / _RPM_LIMIT  # ~1s between calls

_LANG_NAMES = {"fra": "French", "eng": "English"}

_MODEL_CANDIDATES = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def _find_dotenv() -> Path | None:
    """Search for .env up from cwd and from known project locations."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",  # if cwd is py/
        Path(__file__).resolve().parents[3] / ".env",  # source tree
        Path(__file__).resolve().parents[5] / ".env",  # site-packages fallback
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _make_client() -> genai.Client:
    env_path = _find_dotenv()
    if env_path:
        load_dotenv(env_path)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (check .env)")
    return genai.Client(api_key=api_key)


def _resolve_model(client: genai.Client) -> str:
    """Try model candidates in order, return first that exists."""
    for name in _MODEL_CANDIDATES:
        try:
            client.models.get(model=f"models/{name}")
            return name
        except Exception:
            continue
    raise RuntimeError(
        f"None of {_MODEL_CANDIDATES} available. Check model names."
    )


def _build_prompt(sentences: list[str], lang_code: str) -> str:
    lang = _LANG_NAMES.get(lang_code, lang_code)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return (
        f"Paraphrase each of the following {lang} sentences. "
        f"Keep the same meaning but use different words/structure. "
        f"IMPORTANT: The paraphrase MUST remain in {lang} — do NOT translate to another language. "
        "Return one paraphrase per line, numbered to match, in the same order. "
        "Do NOT add explanations.\n\n"
        f"{numbered}"
    )


def _parse_response(text: str, n: int) -> list[str]:
    """Extract n paraphrases from numbered response lines."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # strip leading "1." / "1)" style numbering
    cleaned: list[str] = []
    for line in lines:
        for sep in [". ", ") "]:
            idx = line.find(sep)
            if idx != -1 and line[:idx].strip().isdigit():
                line = line[idx + len(sep) :]
                break
        cleaned.append(line.strip())
    # if we got more than n, truncate; fewer → pad with empty
    return (cleaned[:n] + [""] * n)[:n]


def _paraphrase_batch(
    client: genai.Client,
    model: str,
    sentences: list[str],
    lang_code: str,
) -> list[str]:
    """Call Gemini to paraphrase a batch of sentences."""
    prompt = _build_prompt(sentences, lang_code)
    resp = client.models.generate_content(model=model, contents=prompt)
    return _parse_response(resp.text, len(sentences))


# ---------------------------------------------------------------------------
# Core augmentation
# ---------------------------------------------------------------------------


def augment_df(
    df: pd.DataFrame,
    multiplier: int = 1,
    client: genai.Client | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return augmented rows (only) for mbay train rows.

    Each qualifying row produces `multiplier` paraphrased variants.
    """
    if client is None:
        client = _make_client()
    if model is None:
        model = _resolve_model(client)

    mask = (df["source_lang"] == "mbay") & (df["split"] == "train")
    eligible = df.loc[mask].reset_index(drop=True)

    if verbose:
        print(
            f"Eligible rows: {len(eligible)} | "
            f"Multiplier: {multiplier} | "
            f"Model: {model}"
        )

    new_rows: list[dict] = []
    last_call = 0.0

    for _round in range(multiplier):
        if verbose:
            print(f"  Round {_round + 1}/{multiplier}")

        # Group by target_lang to keep prompts monolingual
        for lang_code, group in eligible.groupby("target_lang"):
            sentences = group["target_text"].tolist()

            for batch_start in range(0, len(sentences), _BATCH_SIZE):
                batch = sentences[batch_start : batch_start + _BATCH_SIZE]
                batch_indices = list(
                    range(batch_start, min(batch_start + _BATCH_SIZE, len(sentences)))
                )

                # Rate limit
                elapsed = time.time() - last_call
                if elapsed < _MIN_INTERVAL:
                    time.sleep(_MIN_INTERVAL - elapsed)

                paraphrases = _paraphrase_batch(client, model, batch, lang_code)
                last_call = time.time()

                for idx, para in zip(batch_indices, paraphrases):
                    if not para:
                        continue
                    row = group.iloc[idx]
                    new_rows.append(
                        {
                            "source_lang": row["source_lang"],
                            "target_lang": row["target_lang"],
                            "source_text": row["source_text"],
                            "target_text": para,
                            "domain": row["domain"],
                            "provenance": f"augment:gemini:{row['provenance']}",
                            "split": "train",
                        }
                    )

                if verbose and batch_start % (10 * _BATCH_SIZE) == 0:
                    total = len(eligible) * multiplier
                    done = _round * len(eligible) + batch_start + len(batch)
                    print(f"    {done}/{total} rows processed")

    return pd.DataFrame(new_rows, columns=df.columns)


# ---------------------------------------------------------------------------
# Sample validation
# ---------------------------------------------------------------------------


def sample_validate(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    n: int = 10,
) -> None:
    """Print n random original→paraphrase pairs for manual inspection."""
    if len(augmented) == 0:
        print("No augmented rows to validate.")
        return

    n = min(n, len(augmented))
    sample = augmented.sample(n, random_state=42)

    for i, (_, aug_row) in enumerate(sample.iterrows(), 1):
        # find original by source_text + target_lang
        orig_prov = aug_row["provenance"].replace("augment:gemini:", "", 1)
        orig_match = original.loc[
            (original["provenance"] == orig_prov)
            & (original["target_lang"] == aug_row["target_lang"])
        ]
        orig_target = orig_match["target_text"].iloc[0] if len(orig_match) else "???"
        print(f"--- [{i}/{n}] ({aug_row['target_lang']}) ---")
        print(f"  Mbay:     {aug_row['source_text']}")
        print(f"  Original: {orig_target}")
        print(f"  Paraph:   {aug_row['target_text']}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paraphrase-augment Mbay parallel corpus via Gemini"
    )
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Number of paraphrase variants per row (default 1)",
    )
    parser.add_argument(
        "--sample-validate",
        type=int,
        default=0,
        help="Print N random pairs for inspection before writing",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Only process first N eligible rows (for testing)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.head > 0:
        # Keep first N mbay+train rows, plus all others unchanged
        mask = (df["source_lang"] == "mbay") & (df["split"] == "train")
        eligible_idx = df.loc[mask].index[: args.head]
        # Temporarily replace the full df with just those rows for augmentation
        aug_source = df.loc[eligible_idx].reset_index(drop=True)
    else:
        aug_source = df

    augmented = augment_df(aug_source, multiplier=args.multiplier, verbose=True)
    print(f"\nGenerated {len(augmented)} augmented rows.")

    if args.sample_validate > 0:
        print(f"\n=== Sample validation ({args.sample_validate} pairs) ===\n")
        sample_validate(aug_source, augmented, n=args.sample_validate)

    # Write: original + augmented
    combined = pd.concat([df, augmented], ignore_index=True)
    combined.to_parquet(args.output, index=False)
    print(f"Written {len(combined)} total rows to {args.output}")


if __name__ == "__main__":
    main()
