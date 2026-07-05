"""SageMaker training entrypoint.

Wraps the NLLB trainer for SageMaker's convention:
- Model artifacts in /opt/ml/model/
- Input data channels in /opt/ml/input/data/{channel}/
- Checkpoints in /opt/ml/checkpoints/
- Hyperparams passed as CLI args by SageMaker
"""

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


def main():
    training_dir = SM_DATA_DIR / "training"
    corpus_file = _get_arg_value("--corpus-file")
    if corpus_file:
        data_path = training_dir / corpus_file
        if not data_path.exists():
            raise FileNotFoundError(f"--corpus-file {corpus_file!r} not found in {training_dir}")
    else:
        # Find parquet file in training channel (supports any corpus version).
        parquets = sorted(training_dir.glob("mbay-corpus-*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No mbay-corpus-*.parquet found in {training_dir}")
        data_path = parquets[-1]  # latest version
    print(f"Using data: {data_path}")
    model_dir = SM_DATA_DIR / "model"

    # Build args for train_nllb.train.main()
    train_args = [
        "train.py",
        "--model-dir", str(model_dir),
        "--data", str(data_path),
        "--output-dir", str(SM_CHECKPOINT_DIR),
    ]

    # Forward all SM hyperparams (passed as --key value by SageMaker).
    # Filter out the ones we already set.
    skip = {"--model-dir", "--data", "--output-dir", "--corpus-file"}
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

    from mbay_nmt.train_nllb.train import main as train_main
    train_main()

    # Copy final model to SM_MODEL_DIR for artifact packaging.
    final_dir = SM_CHECKPOINT_DIR / "final"
    if final_dir.exists():
        shutil.copytree(final_dir, SM_MODEL_DIR, dirs_exist_ok=True)
        print(f"Copied final model to {SM_MODEL_DIR}")


if __name__ == "__main__":
    main()
