import os
import re
import time

PATTERN = re.compile(r"[/_\\]")


def ensure_flash_attention():
    """ensure latest version of flash attention is installed"""
    os.system("pip install flash-attn --no-build-isolation --upgrade")


def training_job_name(model_id: str) -> str:
    job_name = f"mbay-nmt-{model_id}"

    return PATTERN.sub("-", job_name)
