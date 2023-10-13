import os


def ensure_flash_attention():
    """ensure latest version of flash attention is installed"""
    os.system("pip install flash-attn --no-build-isolation --upgrade")

def training_job_name(model_id: str)-> str:
    return f'mbay-nmt-{model_id}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'.replace("/", "-")
