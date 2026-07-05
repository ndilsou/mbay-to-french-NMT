"""Launch a SageMaker MMS ASR adapter training job for Mbay.

Prerequisites, from py/:
    poe build-project sagemaker_train

The existing sagemaker_train project wheel is reused because the root Hatch
config force-includes bases/mbay_nmt into the mbay_nmt package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _prepare_source_dir(project_root: Path) -> Path:
    """Assemble source_dir from pre-built wheel + requirements.txt + ASR entrypoint."""
    project_dir = project_root / "projects" / "sagemaker_train"
    reqs = project_dir / "requirements.txt"
    if not reqs.exists():
        raise FileNotFoundError(f"{reqs} not found. Run: poe build-project sagemaker_train")

    wheels = sorted((project_dir / "dist").glob("mbay_nmt_sagemaker_train-*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheel found in dist/. Run: poe build-project sagemaker_train")
    wheel = wheels[-1]

    staging = project_root / ".sagemaker-asr-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    shutil.copy(project_root / "bases" / "mbay_nmt" / "train_asr" / "entrypoint.py", staging / "entrypoint.py")
    shutil.copy(wheel, staging / wheel.name)

    dlc_provided = {
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "safetensors",
        "tokenizers",
        "sentencepiece",
        "huggingface-hub",
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "pyyaml",
        "tqdm",
        "filelock",
        "packaging",
        "requests",
        "urllib3",
        "certifi",
        "jinja2",
        "markupsafe",
        "sympy",
        "networkx",
        "setuptools",
        "botocore",
        "boto3",
        "s3transfer",
        "s3fs",
        "fsspec",
        "psutil",
        "regex",
        "six",
        "idna",
        "typing-extensions",
        "pydantic",
        "pydantic-core",
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
    parser.add_argument("--s3-manifest", required=True, help="S3 URI for a manifest parquet or directory")
    parser.add_argument("--s3-audio", required=True, help="S3 URI for the audio channel directory")
    parser.add_argument("--s3-output", required=True)
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--instance", default="ml.g5.2xlarge")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=12)
    parser.add_argument("--model-id", default="facebook/mms-1b-all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--max-audio-seconds", type=float, default=20.0)
    parser.add_argument("--min-audio-seconds", type=float, default=0.3)
    parser.add_argument("--kinds", default="sentence,expression,headword")
    parser.add_argument("--train-scope", default="adapter", choices=["adapter", "encoder-top", "full"])
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
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "warmup-ratio": args.warmup_ratio,
        "eval-fraction": args.eval_fraction,
        "max-audio-seconds": args.max_audio_seconds,
        "min-audio-seconds": args.min_audio_seconds,
        "kinds": args.kinds,
        "train-scope": args.train_scope,
    }
    if args.manifest_file:
        hyperparameters["manifest-file"] = args.manifest_file

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
        environment["WANDB_PROJECT"] = "mbay-ASR"

    from sagemaker.huggingface import HuggingFace

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"mbay-mms-asr-{timestamp}"
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

    input_channels = {"manifest": args.s3_manifest, "audio": args.s3_audio}
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

