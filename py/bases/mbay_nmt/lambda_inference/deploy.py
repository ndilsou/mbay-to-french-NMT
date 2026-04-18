"""Deploy Mbay translator Lambda via AWS SAM CLI.

Generates a SAM template and runs sam build + sam deploy. Requires:
- AWS credentials configured
- SAM CLI installed (brew install aws-sam-cli)
- Docker running (for container build)

Usage:
    uv run python -m mbay_nmt.lambda_inference.deploy \\
        --model-dir models/nllb-600m-mbay-int8 \\
        --stack-name mbay-translator \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SAM_TEMPLATE = """\
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Mbay translation API (NLLB-600M INT8)

Globals:
  Function:
    Timeout: 60
    MemorySize: 10240

Resources:
  TranslateFunction:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      Architectures:
        - x86_64
      EphemeralStorage:
        Size: 10240
      Events:
        Translate:
          Type: Api
          Properties:
            Path: /translate
            Method: post
        Health:
          Type: Api
          Properties:
            Path: /health
            Method: get
    Metadata:
      DockerTag: latest
      DockerContext: .
      Dockerfile: Dockerfile

Outputs:
  TranslateApi:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/"
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True, help="Quantized model dir")
    p.add_argument("--stack-name", default="mbay-translator")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    build_dir = Path("bases/mbay_nmt/lambda_inference/.sam-build")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Copy Lambda artifacts to build dir
    src = Path("bases/mbay_nmt/lambda_inference")
    for f in ["handler.py", "Dockerfile", "requirements-lambda.txt"]:
        shutil.copy(src / f, build_dir / f)

    # Copy model
    model_dest = build_dir / "model"
    if model_dest.exists():
        shutil.rmtree(model_dest)
    shutil.copytree(args.model_dir, model_dest)

    # Write SAM template
    (build_dir / "template.yaml").write_text(SAM_TEMPLATE)

    print(f"Build dir: {build_dir}")
    print(f"Model: {args.model_dir}")

    if args.dry_run:
        print("Dry run — skipping sam build/deploy")
        return

    # SAM build
    subprocess.run(
        ["sam", "build", "--template-file", str(build_dir / "template.yaml")],
        check=True,
    )

    # SAM deploy
    subprocess.run(
        [
            "sam", "deploy",
            "--stack-name", args.stack_name,
            "--region", args.region,
            "--capabilities", "CAPABILITY_IAM",
            "--resolve-image-repos",
            "--no-confirm-changeset",
        ],
        check=True,
    )
    print(f"\nDeployed stack: {args.stack_name}")


if __name__ == "__main__":
    main()
