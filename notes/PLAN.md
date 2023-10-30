# Plan

* [X] Create initial pipeline

  * [X] format dataset
  * [X] integrate with Tensorboard or WandB
  * [X] create Sagemaker fine-tuning task script for T5 family
  * [X] Upload dataset to Sagemaker data bucket
  * [X] Create submission notebook
* [ ] Training Script
  * [X] Report predicted samples via WandB see [this](https://docs.wandb.ai/guides/integrations/huggingface#custom-logging-log-and-view-evaluation-samples-during-training)
  * [X] Implement [HF Trainer Early Stopping callback](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback)
  * [X] Add support for Spot instances
  * [ ] Fix OOM issue linked to checkpointing a PEFT model during training. (see [this](https://github.com/huggingface/transformers/issues/23307), [this](https://discuss.huggingface.co/t/peft-lora-gpt-neox-loraconfig/35790), [this](https://github.com/huggingface/peft/issues/96))
* [ ] Train MT5 family:
  * [ ] ~~t5-base~~
  * [ ] ~~google/t5-v1_1-base~~
  * [ ] google/mt5-xl
  * [ ] ~~google/flan-t5-base~~
  * [ ] google/umt5-xl
* [ ] Train ByT5
  * [ ] google/byt5-xl
* [ ] Train M2M100 family: (see [this](https://huggingface.co/tartuNLP/m2m100_418M_smugri) to add custom language to the tokenizer)
  * [ ] facebook/m2m100_1.2B
  * [ ] facebook/m2m100_418M
* [ ] pre-training
  * [ ] prepare pre-training dataset
  * [ ] create Sagemaker pre-trainig task script for T5 family
* [ ] Source bible data
  * [ ] augment dataset with records from the Mbay bible
  * [ ] repeat pre-training
  * [ ] repeat fine-tuning
* [ ] improve dataset quality
  * [X] replace blanks and '^^' with correct linked word.
  * [X] replace blanks and '____' with correct parent entry.
  * [ ] remove duplicates (After lowercasing)
  * [X] correct obviously incorrect traductions
  * [ ] for entries build dataset without the extra context in parenthesis to be consistent with the examples and expressions. -> example and expressions also have this on occasions..
