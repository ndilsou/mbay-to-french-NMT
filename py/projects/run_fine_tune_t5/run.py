from mbay_nmt.training.core import ensure_flash_attention

ensure_flash_attention()

# This is understood to be a hack for SageMaker TrainingJob
from mbay_nmt.fine_tune_t5_family import main  # noqa: E402

if __name__ == "__main__":
    main()
