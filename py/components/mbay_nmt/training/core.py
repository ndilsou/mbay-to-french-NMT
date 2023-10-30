import os
import re
import random
import string
from datasets import Dataset
import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)

from transformers.integrations import WandbCallback
import pandas as pd
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

PATTERN = re.compile(r"[/_\\]")


def ensure_flash_attention():
    """ensure latest version of flash attention is installed"""
    os.system("pip install flash-attn --no-build-isolation --upgrade")


def training_job_name(model_id: str) -> str:
    # Generate a random 5 alphanum string to salt the job name
    salt = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    job_name = f"mbay-nmt-{model_id}-{salt}"

    return PATTERN.sub("-", job_name)


def decode_predictions(tokenizer, predictions, input_ids):
    input_ids = np.where(
        predictions.label_ids != -100, input_ids, tokenizer.pad_token_id
    )
    prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    label_ids = np.where(
        predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id
    )
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    preds = predictions.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    prediction_text = tokenizer.batch_decode(
        preds.argmax(axis=-1), skip_special_tokens=True
    )

    return {"prompt": prompt, "labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback augments the default WandbCallback and logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(
        self,
        trainer: Trainer,
        tokenizer: AutoTokenizer,
        val_dataset: Dataset,
        num_samples=100,
    ):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs

        # generate predictions
        predictions = self.trainer.predict(self.sample_dataset)
        # decode predictions and labels
        predictions = decode_predictions(
            self.tokenizer, predictions, self.sample_dataset["input_ids"]
        )
        # add predictions to a wandb.Table
        predictions_df = pd.DataFrame(predictions)
        predictions_df["epoch"] = state.epoch
        records_table = self._wandb.Table(dataframe=predictions_df)
        print(records_table)
        # log the table to wandb
        self._wandb.log({"sample_predictions": records_table})


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
