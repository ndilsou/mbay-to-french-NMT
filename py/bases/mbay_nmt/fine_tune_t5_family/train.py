from functools import partial
from dataclasses import dataclass, field
from datetime import datetime
import os
import sys
import bitsandbytes as bnb
import torch
import wandb
from huggingface_hub import login
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    SchedulerType,
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_from_disk

from mbay_nmt.training.core import (
    SavePeftModelCallback,
    WandbPredictionProgressCallback,
)

from . import utils

PROJECT_NAME = "mbay-nmt"


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_id: str = field(
        default=None, metadata={"help": "Model id to use for training."}
    )
    dataset_path: str = field(
        default="/opt/ml/input/data/training", metadata={"help": "Path to dataset."}
    )
    warmup_ratio: float = field(default=0.02, metadata={"help": "Warmup ratio"})
    hf_token: str = field(default=None, metadata={"help": "Huggingface token"})
    wandb_token: str = field(default=None, metadata={"help": "WandB token"})
    epochs: int = field(default=3, metadata={"help": "Number of epochs to train for."})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size to use for training."}
    )
    lr: float = field(
        default=5e-5, metadata={"help": "Learning rate to use for training."}
    )
    seed: int = field(default=93100, metadata={"help": "Seed to use for training."})
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Path to deepspeed config file."}
    )
    bf16: bool = field(default=None, metadata={"help": "Whether to use bf16."})
    merge_weights: bool = field(
        default=True,
        metadata={"help": "Whether to merge LoRA weights with base model."},
    )
    output_dir: str = field(
        default="/opt/ml/checkpoints", metadata={"help": "Path to save model."}
    )


def parse_args() -> Arguments:
    """Parse the arguments."""

    parser = HfArgumentParser(Arguments)
    args: Arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        # pre-process the model by upcasting the layer norms in float 32 for
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model


def training_function(args: Arguments):
    run_name = os.environ["TRAINING_JOB_NAME"]
    # configure WanB
    if args.wandb_token:
        wandb.login(key=args.wandb_token)  # Pass your W&B API key here
        wandb.init(
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            resume=True,
        )

    # set seed
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # dataset = utils.tokenize_dataset(tokenizer, dataset)
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False
        if args.gradient_checkpointing
        else True,  # this is needed for gradient checkpointing
        device_map="auto",
        quantization_config=bnb_config,
    )

    # create peft config
    model = create_peft_model(
        model, gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16
    )

    # Define training args
    # output_dir = "/opt/ml/model"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True
        if get_last_checkpoint(args.output_dir) is not None
        else False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # gradient_accumulation_steps=1,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=5,
        lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
        # logging strategies
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        save_strategy="steps",
        # save_steps=500,
        save_steps=50,  # DEBUG
        report_to="wandb" if args.wandb_token else None,
        # report_to="none",  # We manually add our custom wandb callback below.
        run_name=run_name,
        evaluation_strategy="steps",
        # eval_steps=250,
        eval_steps=50, # DEBUG
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator,
        compute_metrics=partial(utils.compute_metrics, tokenizer),
        preprocess_logits_for_metrics=utils.preprocess_logits_for_metrics,
        callbacks=[SavePeftModelCallback],
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=dataset["validation"],
        num_samples=10,
    )
    trainer.add_callback(progress_callback)

    print("Registered callbacks: ", trainer.callback_handler.callbacks)

    # Start training
    # check if checkpoint existing if so continue training
    if get_last_checkpoint(args.output_dir) is not None:
        print("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    sagemaker_save_dir = "/opt/ml/model/"
    if args.merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForSeq2SeqLM

        # load PEFT model in fp16
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(
            args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=False, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(
            sagemaker_save_dir, safe_serialization=False
        )  # XXX: safe_serialization=True causes an error

    # save tokenizer for easy inference
    # tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_args()
    training_function(args)


if __name__ == "__main__":
    main()
