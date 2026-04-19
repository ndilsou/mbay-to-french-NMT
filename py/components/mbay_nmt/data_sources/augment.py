"""Paraphrase augmentation for the Mbay parallel corpus via Gemini Batch API.

Submits all paraphrase requests as a single batch job (no rate limiting,
50% cheaper than real-time). Polls for completion, then parses results.

CLI usage:
    uv run python -m mbay_nmt.data_sources.augment \
        --input datasets/processed/mbay-corpus-v1.parquet \
        --output datasets/processed/mbay-corpus-v1-aug.parquet \
        --multiplier 3

    # Check status of a running batch:
    uv run python -m mbay_nmt.data_sources.augment --status <batch-name>

    # Collect results from a completed batch:
    uv run python -m mbay_nmt.data_sources.augment \
        --collect <batch-name> \
        --input datasets/processed/mbay-corpus-v1.parquet \
        --output datasets/processed/mbay-corpus-v1-aug.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LANG_NAMES = {"fra": "French", "eng": "English"}
_MODEL = "gemini-2.5-flash-lite"
_POLL_INTERVAL = 30  # seconds between status checks

_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def _find_dotenv() -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[5] / ".env",
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


def _build_prompt(sentence: str, lang_code: str) -> str:
    """Build a single-sentence paraphrase prompt."""
    lang = _LANG_NAMES.get(lang_code, lang_code)
    return (
        f"Paraphrase the following {lang} sentence. "
        f"Keep the same meaning but use different words/structure. "
        f"IMPORTANT: The paraphrase MUST remain in {lang}. "
        f"Return ONLY the paraphrased sentence, nothing else.\n\n"
        f"{sentence}"
    )


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def _build_batch_requests(
    df: pd.DataFrame, multiplier: int = 1
) -> tuple[list[dict], list[dict]]:
    """Build inline batch requests for all eligible rows.

    Returns (batch_requests, metadata) where metadata[i] maps to
    batch_requests[i] with the original row info needed to reconstruct output.
    """
    mask = (df["source_lang"] == "mbay") & (df["split"] == "train")
    eligible = df.loc[mask].reset_index(drop=True)

    requests = []
    metadata = []

    for _round in range(multiplier):
        for _, row in eligible.iterrows():
            prompt = _build_prompt(row["target_text"], row["target_lang"])
            requests.append({
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            })
            metadata.append({
                "source_lang": row["source_lang"],
                "target_lang": row["target_lang"],
                "source_text": row["source_text"],
                "domain": row["domain"],
                "provenance": row["provenance"],
                "round": _round,
            })

    return requests, metadata


def submit_batch(
    df: pd.DataFrame,
    multiplier: int = 1,
    client: genai.Client | None = None,
) -> tuple[str, list[dict]]:
    """Submit a batch job and return (batch_name, metadata).

    Saves metadata to a sidecar file so results can be collected later.
    """
    if client is None:
        client = _make_client()

    requests, metadata = _build_batch_requests(df, multiplier)
    print(f"Submitting batch: {len(requests)} requests ({multiplier}x multiplier)")

    batch_job = client.batches.create(
        model=_MODEL,
        src=requests,
        config={"display_name": f"mbay-augment-{multiplier}x"},
    )

    print(f"Batch created: {batch_job.name}")
    print(f"State: {batch_job.state.name}")

    # Save metadata sidecar for later collection
    meta_path = Path(f"/tmp/augment-batch-meta-{batch_job.name.split('/')[-1]}.json")
    meta_path.write_text(json.dumps(metadata))
    print(f"Metadata saved: {meta_path}")

    return batch_job.name, metadata


def check_status(batch_name: str, client: genai.Client | None = None) -> str:
    if client is None:
        client = _make_client()
    job = client.batches.get(name=batch_name)
    print(f"Batch: {batch_name}")
    print(f"State: {job.state.name}")
    return job.state.name


def poll_until_done(
    batch_name: str, client: genai.Client | None = None
) -> str:
    """Poll until batch completes. Returns final state name."""
    if client is None:
        client = _make_client()

    job = client.batches.get(name=batch_name)
    while job.state.name not in _TERMINAL_STATES:
        print(f"  {job.state.name} ... waiting {_POLL_INTERVAL}s")
        time.sleep(_POLL_INTERVAL)
        job = client.batches.get(name=batch_name)

    print(f"Final state: {job.state.name}")
    return job.state.name


def collect_results(
    batch_name: str,
    metadata: list[dict],
    client: genai.Client | None = None,
) -> pd.DataFrame:
    """Collect results from a completed batch job into augmented rows."""
    if client is None:
        client = _make_client()

    job = client.batches.get(name=batch_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch not succeeded: {job.state.name}")

    responses = job.dest.inlined_responses
    if not responses:
        raise RuntimeError("No inline responses found")

    print(f"Collecting {len(responses)} responses...")

    new_rows = []
    errors = 0
    for i, resp in enumerate(responses):
        if resp.response and resp.response.text:
            text = resp.response.text.strip()
            meta = metadata[i]
            new_rows.append({
                "source_lang": meta["source_lang"],
                "target_lang": meta["target_lang"],
                "source_text": meta["source_text"],
                "target_text": text,
                "domain": meta["domain"],
                "provenance": f"augment:gemini:{meta['provenance']}",
                "split": "train",
            })
        else:
            errors += 1

    print(f"Collected {len(new_rows)} paraphrases ({errors} errors)")
    return pd.DataFrame(new_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paraphrase-augment Mbay parallel corpus via Gemini Batch API"
    )
    parser.add_argument("--input", help="Input parquet path")
    parser.add_argument("--output", help="Output parquet path")
    parser.add_argument(
        "--multiplier", type=int, default=1,
        help="Number of paraphrase variants per row (default 1)",
    )
    parser.add_argument(
        "--head", type=int, default=0,
        help="Only process first N eligible rows (for testing)",
    )
    # Batch workflow
    parser.add_argument(
        "--status", metavar="BATCH_NAME",
        help="Check status of a batch job",
    )
    parser.add_argument(
        "--collect", metavar="BATCH_NAME",
        help="Collect results from a completed batch job",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit batch and exit without waiting for results",
    )
    args = parser.parse_args()

    # --- Status check mode ---
    if args.status:
        check_status(args.status)
        return

    # --- Collect mode ---
    if args.collect:
        if not args.input or not args.output:
            parser.error("--collect requires --input and --output")
        batch_id = args.collect.split("/")[-1]
        meta_path = Path(f"/tmp/augment-batch-meta-{batch_id}.json")
        if not meta_path.exists():
            parser.error(f"Metadata file not found: {meta_path}")
        metadata = json.loads(meta_path.read_text())

        augmented = collect_results(args.collect, metadata)
        df = pd.read_parquet(args.input)
        combined = pd.concat([df, augmented], ignore_index=True)
        combined.to_parquet(args.output, index=False)
        print(f"Written {len(combined)} total rows to {args.output}")
        return

    # --- Submit mode (default) ---
    if not args.input or not args.output:
        parser.error("--input and --output are required")

    df = pd.read_parquet(args.input)
    if args.head > 0:
        mask = (df["source_lang"] == "mbay") & (df["split"] == "train")
        eligible_idx = df.loc[mask].index[: args.head]
        aug_source = df.loc[eligible_idx].reset_index(drop=True)
    else:
        aug_source = df

    client = _make_client()
    batch_name, metadata = submit_batch(
        aug_source, multiplier=args.multiplier, client=client,
    )

    if args.no_wait:
        print(f"\nBatch submitted. Check later with:\n"
              f"  uv run python -m mbay_nmt.data_sources.augment --status {batch_name}")
        return

    # Poll and collect
    state = poll_until_done(batch_name, client)
    if state != "JOB_STATE_SUCCEEDED":
        print(f"Batch failed: {state}", file=sys.stderr)
        sys.exit(1)

    augmented = collect_results(batch_name, metadata, client)
    combined = pd.concat([df, augmented], ignore_index=True)
    combined.to_parquet(args.output, index=False)
    print(f"\nWritten {len(combined)} total rows to {args.output}")


if __name__ == "__main__":
    main()
