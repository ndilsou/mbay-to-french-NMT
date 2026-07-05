"""Launch a SageMaker training job for the Mbay ByT5 tone restorer.

Prerequisites, from py/:
    poe build-project tone_restore_train
    uv run python -m mbay_nmt.tone_restore.prep_byt5_pairs
    aws s3 cp datasets/processed/tone-restore-pairs.parquet s3://<bucket>/mbay-tone/pairs/

Uses its own Polylith project (projects/tone_restore_train) and staging dir so
the sagemaker_train project used by the NLLB/ASR jobs is never touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _prepare_source_dir(project_root: Path) -> Path:
    project_dir = project_root / "projects" / "tone_restore_train"
    reqs = project_dir / "requirements.txt"
    if not reqs.exists():
        raise FileNotFoundError(f"{reqs} not found. Run: poe build-project tone_restore_train")

    wheels = sorted((project_dir / "dist").glob("mbay_nmt_tone_restore_train-*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheel found in dist/. Run: poe build-project tone_restore_train")
    wheel = wheels[-1]

    staging = project_root / ".sagemaker-tone-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    shutil.copy(
        project_root / "bases" / "mbay_nmt" / "train_tone_restore" / "entrypoint.py",
        staging / "entrypoint.py",
    )
    shutil.copy(wheel, staging / wheel.name)

    dlc_provided = {
        "torch", "transformers", "datasets", "accelerate", "peft",
        "safetensors", "tokenizers", "sentencepiece", "huggingface-hub",
        "numpy", "scipy", "pandas", "pyarrow", "pyyaml", "tqdm",
        "filelock", "packaging", "requests", "urllib3", "certifi",
        "jinja2", "markupsafe", "sympy", "networkx", "setuptools",
        "botocore", "boto3", "s3transfer", "s3fs", "fsspec",
        "psutil", "regex", "six", "idna",
        "typing-extensions", "pydantic", "pydantic-core",
    }
    normalized_dlc = {name.replace("-", "_") for name in dlc_provided}
    filtered_lines = []
    for line in reqs.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(" "):
            continue
        pkg_name = stripped.split("==")[0].split(">=")[0].split(";")[0].strip().lower()
        if pkg_name.startswith(("nvidia-", "cuda-")) or pkg_name in {"triton", "bitsandbytes"}:
            continue
        if pkg_name.replace("-", "_") not in normalized_dlc:
            filtered_lines.append(stripped)

    (staging / "requirements.txt").write_text("\n".join(filtered_lines) + f"\n{wheel.name}\n")
    print(f"Source dir: {staging}")
    print(f"  wheel: {wheel.name}")
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-pairs", required=True, help="S3 URI of dir containing the pairs parquet")
    parser.add_argument("--s3-output", required=True)
    parser.add_argument("--pairs-file", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--instance", default="ml.g5.xlarge")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=4)
    parser.add_argument("--model-id", default="google/byt5-small")
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import sagemaker

    project_root = Path.cwd()
    if not (project_root / "workspace.toml").exists():
        raise SystemExit("Run from the py/ directory (workspace root)")

    source_dir = _prepare_source_dir(project_root)
    session = sagemaker.Session()
    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            import boto3

            identity = boto3.client("sts").get_caller_identity()
            raise SystemExit(
                "Not in a SageMaker notebook. Pass --role explicitly.\n"
                f"Current identity: {identity['Arn']}"
            )

    hyperparameters = {
        "model-id": args.model_id,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "grad-accum": args.grad_accum,
        "lr": args.lr,
        "eval-steps": args.eval_steps,
    }
    if args.pairs_file:
        hyperparameters["pairs-file"] = args.pairs_file

    environment = {}
    if args.use_wandb:
        import netrc

        try:
            auth = netrc.netrc().authenticators("api.wandb.ai")
        except FileNotFoundError:
            auth = None
        if not auth:
            raise SystemExit("--use-wandb requires ~/.netrc with api.wandb.ai credentials")
        environment["WANDB_API_KEY"] = auth[2]
        environment["WANDB_PROJECT"] = "mbay-tone-restore"

    from sagemaker.huggingface import HuggingFace

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"mbay-tone-byt5-{timestamp}"
    checkpoint_s3 = args.s3_output.rstrip("/") + "/checkpoints/"
    estimator = HuggingFace(
        entry_point="entrypoint.py",
        source_dir=str(source_dir),
        role=role,
        instance_count=1,
        instance_type=args.instance,
        use_spot_instances=args.spot,
        max_wait=args.max_run_hours * 3600 + 1800 if args.spot else None,
        max_run=args.max_run_hours * 3600,
        output_path=args.s3_output,
        checkpoint_s3_uri=checkpoint_s3 if args.spot else None,
        transformers_version="4.49.0",
        pytorch_version="2.5.1",
        py_version="py311",
        hyperparameters=hyperparameters,
        environment=environment,
        sagemaker_session=session,
    )

    input_channels = {"pairs": args.s3_pairs}
    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Job name: {job_name}")
        print(f"Instance: {args.instance} (spot={args.spot})")
        print(f"Source dir: {source_dir}")
        print(f"Hyperparameters:\n{json.dumps(hyperparameters, indent=2)}")
        print(f"Input channels:\n{json.dumps(input_channels, indent=2)}")
        print(f"Output: {args.s3_output}")
        return

    estimator.fit(input_channels, job_name=job_name)
    print(f"Training job completed: {job_name}")


if __name__ == "__main__":
    main()
