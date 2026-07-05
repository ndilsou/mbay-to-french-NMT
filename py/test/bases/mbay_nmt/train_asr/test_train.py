import unicodedata

import numpy as np

from mbay_nmt.train_asr.train import (
    AsrExample,
    DataCollatorCTCWithPadding,
    build_vocab_dict,
    make_processor,
    normalize_text_for_ctc,
    split_by_text,
)


def test_vocab_builder_round_trips_toned_chars_through_tokenizer():
    texts = ["àbḭ̀yō̰o̰ ɓāl ɗèē ŋgɨ̄"]
    vocab = build_vocab_dict(texts)
    processor = make_processor(vocab)

    nfd_text = normalize_text_for_ctc(texts[0])
    ids = processor.tokenizer(nfd_text).input_ids
    decoded = processor.tokenizer.decode(ids, group_tokens=False)

    assert unicodedata.normalize("NFC", decoded) == unicodedata.normalize("NFC", texts[0])
    assert "\u0300" in vocab
    assert any(unicodedata.combining(char) for char in vocab)
    assert "ḭ" not in vocab


def test_data_collator_pads_audio_and_masks_label_padding():
    processor = make_processor(build_vocab_dict(["aba", "a"]))
    collator = DataCollatorCTCWithPadding(processor)

    batch = collator(
        [
            {"input_values": np.ones(5, dtype=np.float32), "labels": processor.tokenizer("aba").input_ids},
            {"input_values": np.ones(3, dtype=np.float32), "labels": processor.tokenizer("a").input_ids},
        ]
    )

    assert tuple(batch["input_values"].shape) == (2, 5)
    assert tuple(batch["attention_mask"].shape) == (2, 5)
    assert tuple(batch["labels"].shape) == (2, 3)
    assert batch["labels"][1, 1:].tolist() == [-100, -100]


def test_split_by_text_keeps_duplicate_text_out_of_both_splits(tmp_path):
    examples = [
        AsrExample(text="same text", audio_path=tmp_path / "a.mp3", s3_key="a.mp3", kind="sentence"),
        AsrExample(text="same text", audio_path=tmp_path / "b.mp3", s3_key="b.mp3", kind="sentence"),
        AsrExample(text="other text", audio_path=tmp_path / "c.mp3", s3_key="c.mp3", kind="sentence"),
        AsrExample(text="more text", audio_path=tmp_path / "d.mp3", s3_key="d.mp3", kind="sentence"),
    ]

    train, eval_ = split_by_text(examples, eval_fraction=0.5, seed=7)

    train_texts = {normalize_text_for_ctc(example.text) for example in train}
    eval_texts = {normalize_text_for_ctc(example.text) for example in eval_}
    assert train_texts.isdisjoint(eval_texts)
    assert len(train) + len(eval_) == len(examples)
