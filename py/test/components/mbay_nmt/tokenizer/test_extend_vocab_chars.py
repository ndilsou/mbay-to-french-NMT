import pandas as pd
import torch

from mbay_nmt.tokenizer.extend_vocab_chars import add_legit_chars, base_char, build_char_inventory


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab = {"<unk>": 0, "a": 1, "i": 2, "w": 3, "y": 4, " ": 5}
        self.special_flags: list[bool] = []

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=False):
        if text == "aẁa":
            return {"input_ids": [self.vocab["a"], self.unk_token_id, self.vocab["a"]]}
        ids = []
        for char in text:
            if char in {"ḭ", "ȳ", "Ɂ", "╬"}:
                ids.append(self.unk_token_id)
            else:
                ids.append(self.vocab.get(char, 10 + ord(char) % 1000))
        return {"input_ids": ids}

    def add_tokens(self, tokens, special_tokens=False):
        self.special_flags.append(special_tokens)
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class FakeModel:
    def __init__(self, vocab_size):
        self.input_embedding = torch.nn.Embedding(vocab_size, 3)
        self.output_embedding = torch.nn.Embedding(vocab_size, 3)
        with torch.no_grad():
            self.input_embedding.weight.copy_(torch.arange(vocab_size * 3, dtype=torch.float32).reshape(vocab_size, 3))
            self.output_embedding.weight.copy_(self.input_embedding.weight)
        self.config = type("Config", (), {"vocab_size": vocab_size})()

    def resize_token_embeddings(self, new_size, mean_resizing=False):
        old_input = self.input_embedding.weight.detach().clone()
        old_output = self.output_embedding.weight.detach().clone()
        self.input_embedding = torch.nn.Embedding(new_size, old_input.shape[1])
        self.output_embedding = torch.nn.Embedding(new_size, old_output.shape[1])
        with torch.no_grad():
            self.input_embedding.weight[: old_input.shape[0]] = old_input
            self.output_embedding.weight[: old_output.shape[0]] = old_output
            self.input_embedding.weight[old_input.shape[0] :] = 0
            self.output_embedding.weight[old_output.shape[0] :] = 0

    def get_input_embeddings(self):
        return self.input_embedding

    def get_output_embeddings(self):
        return self.output_embedding


def test_base_char_strips_diacritics_from_composed_mbay_letters():
    assert base_char("ḭ") == "i"
    assert base_char("ẁ") == "w"
    assert base_char("ȳ") == "y"


def test_build_char_inventory_splits_legit_orthography_from_rare_artifacts():
    rows = [
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "ḭ dictionary",
            "target_text": "gloss",
            "domain": "lexicon",
        },
        {
            "source_lang": "beboto",
            "target_lang": "eng",
            "source_text": "ȳ dictionary",
            "target_text": "gloss",
            "domain": "lexicon",
        },
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "Ɂ artifact",
            "target_text": "gloss",
            "domain": "bible",
        },
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "╬ dictionary symbol",
            "target_text": "gloss",
            "domain": "lexicon",
        },
    ]
    rows.extend(
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "ẁ row",
            "target_text": "gloss",
            "domain": "bible",
        }
        for _ in range(101)
    )
    df = pd.DataFrame(rows)

    unsafe = {item.char: item for item in build_char_inventory(df, FakeTokenizer()) if item.is_unsafe}

    assert unsafe["ḭ"].is_legit_orthography
    assert unsafe["ȳ"].is_legit_orthography
    assert unsafe["ẁ"].is_legit_orthography
    assert not unsafe["Ɂ"].is_legit_orthography
    assert not unsafe["╬"].is_legit_orthography


def test_add_legit_chars_adds_normal_tokens_and_initializes_from_base_embedding():
    tokenizer = FakeTokenizer()
    model = FakeModel(len(tokenizer))
    base_i_vector = model.get_input_embeddings().weight[tokenizer.convert_tokens_to_ids("i")].detach().clone()

    added = add_legit_chars(tokenizer, model, ["ḭ"])
    new_id = tokenizer.convert_tokens_to_ids("ḭ")

    assert added == ["ḭ"]
    assert tokenizer.special_flags == [False]
    assert torch.equal(model.get_input_embeddings().weight[new_id], base_i_vector)
    assert torch.equal(model.get_output_embeddings().weight[new_id], base_i_vector)


def test_add_legit_chars_uses_global_mean_when_base_was_unknown_before_add():
    tokenizer = FakeTokenizer()
    model = FakeModel(len(tokenizer))
    old_mean = model.get_input_embeddings().weight.detach().mean(dim=0).clone()

    added = add_legit_chars(tokenizer, model, ["Ɂ"])
    new_id = tokenizer.convert_tokens_to_ids("Ɂ")

    assert added == ["Ɂ"]
    assert torch.equal(model.get_input_embeddings().weight[new_id], old_mean)
