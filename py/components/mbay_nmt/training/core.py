import os


def ensure_flash_attention():
    """ensure latest version of flash attention is installed"""
    os.system("pip install flash-attn --no-build-isolation --upgrade")
