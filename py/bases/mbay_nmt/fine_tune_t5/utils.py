from datasets import Dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
from typing import Iterable, Literal
from functools import partial

MAX_SEQ_LENGTH = 512

metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(tokenizer: AutoTokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


Lang = Literal["mbay", "french", "english"]


def prepare_pair(examples, prefix: str, source_lang: Lang, target_lang: Lang):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    return inputs, targets


def preprocess_records(tokenizer: AutoTokenizer, examples, padding_max_length=True):
    inputs: list[str] = []
    targets: list[str] = []

    _inputs, _target = prepare_pair(
        examples, "Translate English to Mbay: ", "english", "mbay"
    )
    inputs.extend(_inputs)
    targets.extend(_target)

    _inputs, _target = prepare_pair(
        examples, "Translate Mbay to English: ", "mbay", "english"
    )
    inputs.extend(_inputs)
    targets.extend(_target)

    _inputs, _target = prepare_pair(
        examples, "Translate French to Mbay: ", "french", "mbay"
    )
    inputs.extend(_inputs)
    targets.extend(_target)

    _inputs, _target = prepare_pair(
        examples, "Translate Mbay to French: ", "mbay", "french"
    )
    inputs.extend(_inputs)
    targets.extend(_target)

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length" if padding_max_length else None,
    )
    return model_inputs




def tokenize_dataset(tokenizer: AutoTokenizer, dataset: Dataset):
    return dataset.map(
        partial(preprocess_records, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
