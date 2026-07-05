"""Launch a SageMaker training job for Mbay NLLB.

Prerequisites — run from the py/ directory:
    poe build-project sagemaker_train

Usage:
    uv run python -m mbay_nmt.sagemaker_train.launch \\
        --s3-data s3://bucket/mbay-nmt/data/ \\
        --s3-model s3://bucket/mbay-nmt/model-extended/ \\
        --s3-output s3://bucket/mbay-nmt/output/ \\
        [--instance ml.g5.xlarge] [--spot] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _prepare_source_dir(project_root: Path) -> Path:
    """Assemble source_dir from pre-built wheel + requirements.txt + entrypoint."""
    project_dir = project_root / "projects" / "sagemaker_train"
    reqs = project_dir / "requirements.txt"
    if not reqs.exists():
        raise FileNotFoundError(
            f"{reqs} not found. Run: poe build-project sagemaker_train"
        )

    # Find the wheel in dist/
    wheels = sorted((project_dir / "dist").glob("mbay_nmt_sagemaker_train-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            "No wheel found in dist/. Run: poe build-project sagemaker_train"
        )
    wheel = wheels[-1]

    staging = project_root / ".sagemaker-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # Copy entrypoint + wheel + requirements
    shutil.copy(
        project_root / "bases" / "mbay_nmt" / "sagemaker_train" / "entrypoint.py",
        staging / "entrypoint.py",
    )
    shutil.copy(wheel, staging / wheel.name)

    # Write requirements: only our wheel + deps NOT pre-installed in HF DLC.
    # The DLC already has transformers, torch, datasets, accelerate, peft, etc.
    # Installing newer versions breaks torchvision compatibility.
    DLC_PROVIDED = {
        "torch", "transformers", "datasets", "accelerate", "peft",
        "safetensors", "tokenizers", "sentencepiece", "huggingface-hub",
        "numpy", "scipy", "pandas", "pyarrow", "pyyaml", "tqdm",
        "filelock", "packaging", "requests", "urllib3", "certifi",
        "jinja2", "markupsafe", "sympy", "networkx", "setuptools",
        "botocore", "boto3", "s3transfer", "s3fs", "fsspec",
        "psutil", "regex", "six", "idna",
        "typing-extensions", "pydantic", "pydantic-core",
    }
    filtered_lines = []
    for line in reqs.read_text().splitlines():
        # Skip comments and blank lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(" "):
            continue
        pkg_name = stripped.split("==")[0].split(">=")[0].split(";")[0].strip().lower()
        # GPU stack must come from the DLC, never pip (CUDA-13 wheels would
        # shadow the image's CUDA-12 libs).
        if pkg_name.startswith(("nvidia-", "cuda-")) or pkg_name in {"triton", "bitsandbytes"}:
            continue
        if pkg_name.replace("-", "_") not in {p.replace("-", "_") for p in DLC_PROVIDED}:
            filtered_lines.append(stripped)

    (staging / "requirements.txt").write_text(
        "\n".join(filtered_lines) + f"\n{wheel.name}\n"
    )

    print(f"Source dir: {staging}")
    print(f"  wheel: {wheel.name}")
    return staging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s3-data", required=True,
                   help="S3 URI of directory containing the parquet file directly (not a parent dir)")
    p.add_argument("--s3-model", required=True)
    p.add_argument("--s3-output", required=True)
    p.add_argument("--corpus-file", default=None, help="Exact parquet filename within the training channel")
    p.add_argument("--image-uri", default=None, help="ECR image URI (skip HF estimator)")
    p.add_argument("--role", default=None, help="IAM role ARN")
    p.add_argument("--instance", default="ml.g5.xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--max-run-hours", type=int, default=6)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--chrf-eval-every", type=int, default=1)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import sagemaker
    from sagemaker.estimator import Estimator

    # Find workspace root (where workspace.toml lives)
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
            sts = boto3.client("sts")
            identity = sts.get_caller_identity()
            raise SystemExit(
                f"Not in a SageMaker notebook. Pass --role explicitly.\n"
                f"Current identity: {identity['Arn']}"
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"mbay-nllb-{timestamp}"

    hyperparameters = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "grad-accum": args.grad_accum,
        "lr": args.lr,
        "bf16": True,
        "gradient-checkpointing": True,
        "eval-steps": 500,
        "save-steps": 500,
        "early-stopping-patience": 5,
        "label-smoothing": args.label_smoothing,
        "chrf-eval-every": args.chrf_eval_every,
        "use-wandb": args.use_wandb,
    }
    if args.corpus_file:
        hyperparameters["corpus-file"] = args.corpus_file

    environment = {}
    if args.use_wandb:
        import netrc
        try:
            auth = netrc.netrc().authenticators("api.wandb.ai")
            if auth:
                environment["WANDB_API_KEY"] = auth[2]
                environment["WANDB_PROJECT"] = "mbay-NMT"
            else:
                raise SystemExit("--use-wandb requires WANDB_API_KEY. Not found in ~/.netrc")
        except FileNotFoundError:
            raise SystemExit("--use-wandb requires ~/.netrc with api.wandb.ai credentials")

    checkpoint_s3 = args.s3_output.rstrip("/") + "/checkpoints/"

    if args.image_uri:
        estimator = Estimator(
            image_uri=args.image_uri,
            role=role,
            instance_count=1,
            instance_type=args.instance,
            use_spot_instances=args.spot,
            max_wait=args.max_run_hours * 3600 + 1800 if args.spot else None,
            max_run=args.max_run_hours * 3600,
            output_path=args.s3_output,
            checkpoint_s3_uri=checkpoint_s3 if args.spot else None,
            hyperparameters=hyperparameters,
            environment=environment,
            sagemaker_session=session,
        )
    else:
        from sagemaker.huggingface import HuggingFace

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

    input_channels = {
        "training": args.s3_data,
        "model": args.s3_model,
    }

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
