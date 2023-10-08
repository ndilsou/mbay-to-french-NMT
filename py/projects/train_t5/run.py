import os

# upgrade flash attention here
os.system("pip install flash-attn --no-build-isolation --upgrade")

# This is understood to be a hack for SageMaker TrainingJob
from mbay_nmt.fine_tune_t5 import main  # noqa: E402

if __name__ == "__main__":
    main()
