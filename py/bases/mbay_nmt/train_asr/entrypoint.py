"""SageMaker entrypoint for Mbay MMS ASR adapter training."""

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
    manifest_dir = SM_DATA_DIR / "manifest"
    audio_dir = SM_DATA_DIR / "audio"
    manifest_file = _get_arg_value("--manifest-file")
    if manifest_file:
        manifest_path = manifest_dir / manifest_file
        if not manifest_path.exists():
            raise FileNotFoundError(f"--manifest-file {manifest_file!r} not found in {manifest_dir}")
    else:
        parquets = sorted(manifest_dir.glob("*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No parquet manifest found in {manifest_dir}")
        manifest_path = parquets[-1]

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio channel not found: {audio_dir}")

    train_args = [
        "train.py",
        "--manifest",
        str(manifest_path),
        "--audio-dir",
        str(audio_dir),
        "--output-dir",
        str(SM_CHECKPOINT_DIR),
    ]

    skip = {"--manifest", "--audio-dir", "--output-dir", "--manifest-file"}
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
    from mbay_nmt.train_asr.train import main as train_main

    train_main()

    final_dir = SM_CHECKPOINT_DIR / "final"
    if final_dir.exists():
        shutil.copytree(final_dir, SM_MODEL_DIR, dirs_exist_ok=True)
        print(f"Copied final ASR model to {SM_MODEL_DIR}")


if __name__ == "__main__":
    main()

