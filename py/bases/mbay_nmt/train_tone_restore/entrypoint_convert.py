"""SageMaker entrypoint for the bible->Keegan conversion job."""

from __future__ import annotations

import sys
from pathlib import Path

SM_MODEL_DIR = Path("/opt/ml/model")
SM_DATA_DIR = Path("/opt/ml/input/data")


def main() -> None:
    corpus_dir = SM_DATA_DIR / "corpus"
    parquets = sorted(corpus_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet found in {corpus_dir}")

    convert_args = [
        "convert.py",
        str(SM_DATA_DIR / "model1"),
        "--agreement-model",
        str(SM_DATA_DIR / "model2"),
        "--corpus",
        str(parquets[-1]),
        "--output",
        str(SM_MODEL_DIR / "bible-restored.parquet"),
    ]
    passthrough = {"--batch-size", "--num-beams", "--limit"}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in passthrough and i + 1 < len(sys.argv):
            convert_args.extend([arg, sys.argv[i + 1]])
            i += 2
        else:
            i += 1

    sys.argv = convert_args
    from mbay_nmt.train_tone_restore.convert import main as convert_main

    convert_main()


if __name__ == "__main__":
    main()
