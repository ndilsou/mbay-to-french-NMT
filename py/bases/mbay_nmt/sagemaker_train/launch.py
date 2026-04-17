"""Launch a SageMaker training job for Mbay NLLB.

Usage:
    uv run python -m mbay_nmt.sagemaker_train.launch \\
        --s3-data s3://your-bucket/mbay-nmt/data/ \\
        --s3-model s3://your-bucket/mbay-nmt/model-extended/ \\
        --s3-output s3://your-bucket/mbay-nmt/output/ \\
        [--instance ml.g5.xlarge] [--spot] [--dry-run]

Before running:
    1. Upload mbay-corpus-v1.parquet to s3-data as training/mbay-corpus-v1.parquet
    2. Upload models/nllb-600m-mbay-extended/ to s3-model
    3. Build + push Docker image (or use the provided Dockerfile with ECR)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s3-data", required=True, help="S3 path to training data dir")
    p.add_argument("--s3-model", required=True, help="S3 path to extended model dir")
    p.add_argument("--s3-output", required=True, help="S3 output path for artifacts")
    p.add_argument("--image-uri", default=None, help="ECR image URI (if not using SM built-in)")
    p.add_argument("--instance", default="ml.g5.xlarge")
    p.add_argument("--spot", action="store_true", help="Use spot instances (~70% cheaper)")
    p.add_argument("--max-run-hours", type=int, default=6)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print config without launching")
    args = p.parse_args()

    import sagemaker
    from sagemaker.estimator import Estimator

    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
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

    # Use custom Docker image if provided, else SageMaker HuggingFace estimator.
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
            source_dir="bases/mbay_nmt/sagemaker_train",
            role=role,
            instance_count=1,
            instance_type=args.instance,
            use_spot_instances=args.spot,
            max_wait=args.max_run_hours * 3600 + 1800 if args.spot else None,
            max_run=args.max_run_hours * 3600,
            output_path=args.s3_output,
            transformers_version="4.46.0",
            pytorch_version="2.5.0",
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
        print(f"Hyperparameters:\n{json.dumps(hyperparameters, indent=2)}")
        print(f"Input channels:\n{json.dumps(input_channels, indent=2)}")
        print(f"Output: {args.s3_output}")
        return

    estimator.fit(input_channels, job_name=job_name)
    print(f"Training job submitted: {job_name}")


if __name__ == "__main__":
    main()
