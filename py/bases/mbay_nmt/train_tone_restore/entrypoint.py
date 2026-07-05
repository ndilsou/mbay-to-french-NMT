"""SageMaker entrypoint for the Mbay ByT5 tone restorer."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SM_MODEL_DIR = Path("/opt/ml/model")
SM_DATA_DIR = Path("/opt/ml/input/data")
SM_CHECKPOINT_DIR = Path("/opt/ml/checkpoints")


def _get_arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv) or sys.argv[idx + 1].startswith("--"):
        raise ValueError(f"{name} requires a value")
    return sys.argv[idx + 1]


def main() -> None:
    pairs_dir = SM_DATA_DIR / "pairs"
    pairs_file = _get_arg_value("--pairs-file")
    if pairs_file:
        pairs_path = pairs_dir / pairs_file
        if not pairs_path.exists():
            raise FileNotFoundError(f"--pairs-file {pairs_file!r} not found in {pairs_dir}")
    else:
        parquets = sorted(pairs_dir.glob("*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No parquet found in {pairs_dir}")
        pairs_path = parquets[-1]

    train_args = [
        "train.py",
        "--pairs",
        str(pairs_path),
        "--output-dir",
        str(SM_CHECKPOINT_DIR),
    ]

    skip = {"--pairs", "--output-dir", "--pairs-file"}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in skip:
            i += 2
            continue
        train_args.append(arg)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            train_args.append(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    sys.argv = train_args
    from mbay_nmt.train_tone_restore.train import main as train_main

    train_main()

    best_dir = SM_CHECKPOINT_DIR / "best"
    if best_dir.exists():
        # spot checkpoint sync drops transient marker files mid-copy
        shutil.copytree(
            best_dir,
            SM_MODEL_DIR,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.sagemaker-uploading", "*.sagemaker-uploaded"),
        )
        print(f"Copied best tone-restore model to {SM_MODEL_DIR}")


if __name__ == "__main__":
    main()
