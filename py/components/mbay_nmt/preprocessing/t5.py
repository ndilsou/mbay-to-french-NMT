from transformers import AutoTokenizer
from typing import Literal

MAX_SEQ_LENGTH = 512


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
