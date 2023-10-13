# Plan

* [ ] Create initial pipeline
  * [X] format dataset
  * [X] integrate with Tensorboard or WandB
  * [X] create Sagemaker fine-tuning task script for T5 family
  * [ ] Upload dataset to Sagemaker data bucket
  * [X] Create submission notebook
* [ ] Train T5 family:
  * [ ] t5-base
  * [ ] google/t5-v1_1-base
  * [ ] google/mt5-base
  * [ ] google/flan-t5-base
  * [ ] google/umt5-base
* [ ] pre-training
  * [ ] prepare pre-training dataset
  * [ ] create Sagemaker pre-trainig task script for T5 family
* [ ] Source bible data
  * [ ] augment dataset with records from the Mbay bible
  * [ ] repeat pre-training
  * [ ] repeat fine-tuning
* [ ] improve dataset quality
  * [ ] replace blanks and '^^' with correct linked word.
  * [ ] correct obviously incorrect traductions
  * [ ] for entries build dataset without the extra context in parenthesis to be consistent with the examples and expressions.
