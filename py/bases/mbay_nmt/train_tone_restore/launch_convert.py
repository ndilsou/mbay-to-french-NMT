"""Launch the bible->Keegan conversion as a SageMaker job.

Prerequisites, from py/:
    poe build-project tone_restore_train
    aws s3 sync py/models/tone-restore-byt5-base-sm s3://<bucket>/mbay-tone/models/v2/
    aws s3 sync py/models/tone-restore-byt5-sm s3://<bucket>/mbay-tone/models/v1/
    aws s3 cp datasets/processed/mbay-corpus-v2.parquet s3://<bucket>/mbay-tone/corpus/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from mbay_nmt.train_tone_restore.launch import _prepare_source_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-corpus", required=True)
    parser.add_argument("--s3-model1", required=True, help="primary (byt5-base) model dir")
    parser.add_argument("--s3-model2", required=True, help="agreement (byt5-small) model dir")
    parser.add_argument("--s3-output", required=True)
    parser.add_argument("--role", default=None)
    parser.add_argument("--instance", default="ml.g5.xlarge")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import sagemaker

    project_root = Path.cwd()
    if not (project_root / "workspace.toml").exists():
        raise SystemExit("Run from the py/ directory (workspace root)")

    source_dir = _prepare_source_dir(project_root)
    # conversion entrypoint instead of the training one
    import shutil

    shutil.copy(
        project_root / "bases" / "mbay_nmt" / "train_tone_restore" / "entrypoint_convert.py",
        source_dir / "entrypoint_convert.py",
    )

    session = sagemaker.Session()
    if args.role:
        role = args.role
    else:
        role = sagemaker.get_execution_role()

    hyperparameters = {"batch-size": args.batch_size, "num-beams": args.num_beams}
    if args.limit:
        hyperparameters["limit"] = args.limit

    from sagemaker.huggingface import HuggingFace

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"mbay-tone-convert-{timestamp}"
    estimator = HuggingFace(
        entry_point="entrypoint_convert.py",
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
        "corpus": args.s3_corpus,
        "model1": args.s3_model1,
        "model2": args.s3_model2,
    }
    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Job name: {job_name}")
        print(f"Hyperparameters:\n{json.dumps(hyperparameters, indent=2)}")
        print(f"Input channels:\n{json.dumps(input_channels, indent=2)}")
        return

    estimator.fit(input_channels, job_name=job_name)
    print(f"Conversion job completed: {job_name}")


if __name__ == "__main__":
    main()
