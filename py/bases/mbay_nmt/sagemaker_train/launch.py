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

    # Write requirements that installs the wheel (SageMaker runs pip install -r)
    (staging / "requirements.txt").write_text(
        reqs.read_text() + f"\n{wheel.name}\n"
    )

    print(f"Source dir: {staging}")
    print(f"  wheel: {wheel.name}")
    return staging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s3-data", required=True)
    p.add_argument("--s3-model", required=True)
    p.add_argument("--s3-output", required=True)
    p.add_argument("--image-uri", default=None, help="ECR image URI (skip HF estimator)")
    p.add_argument("--role", default=None, help="IAM role ARN")
    p.add_argument("--instance", default="ml.g5.xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--max-run-hours", type=int, default=6)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
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
        "use-wandb": args.use_wandb,
    }

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
            hyperparameters=hyperparameters,
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
            transformers_version="4.49.0",
            pytorch_version="2.5.1",
            py_version="py311",
            hyperparameters=hyperparameters,
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
