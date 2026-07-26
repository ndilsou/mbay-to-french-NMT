"""Extend an existing Mbay NLLB model/tokenizer with tokenizer-unsafe corpus chars.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache-fix UV_NO_SYNC=1 PYTHONPATH=components:bases \
      uv run --no-sync python -m mbay_nmt.tokenizer.extend_vocab_chars \
        --model-dir models/nllb-600m-mbay-extended \
        --corpus datasets/processed/mbay-corpus-v3.2.parquet \
        --output-dir /tmp/nllb-600m-mbay-extended-chars
"""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import regex

from mbay_nmt.data_sources.build_v3 import DICTIONARY_DOMAINS, nfc_strip


@dataclass(frozen=True)
class CharInventory:
    char: str
    rows: int
    occurrences: int
    in_dictionary_text: bool
    direct_unk: bool
    context_unk: bool

    @property
    def is_unsafe(self) -> bool:
        return self.direct_unk or self.context_unk

    @property
    def is_legit_orthography(self) -> bool:
        return is_orthographic_char(self.char) and (self.in_dictionary_text or self.rows > 100)


def is_orthographic_char(char: str) -> bool:
    return all(unicodedata.category(ch)[0] in {"L", "M"} for ch in char)


def _ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def contains_unk(tokenizer: Any, text: str) -> bool:
    return tokenizer.unk_token_id in _ids(tokenizer, text)


def base_char(char: str) -> str:
    decomposed = unicodedata.normalize("NFD", char)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def iter_normalized_texts(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for column in ["source_text", "target_text"]:
        texts.extend(nfc_strip(value) for value in df[column])
    return texts


def dictionary_orthography_chars(df: pd.DataFrame) -> set[str]:
    chars: set[str] = set()
    dictionary_mask = df["domain"].isin(DICTIONARY_DOMAINS)
    for lang_col, text_col in [("source_lang", "source_text"), ("target_lang", "target_text")]:
        orthography_mask = dictionary_mask & ~df[lang_col].isin({"eng", "fra"})
        for text in df.loc[orthography_mask, text_col]:
            chars.update(nfc_strip(text))
    return chars


def build_char_inventory(df: pd.DataFrame, tokenizer: Any) -> list[CharInventory]:
    row_counts: dict[str, int] = {}
    occurrence_counts: dict[str, int] = {}
    for row in df[["source_text", "target_text"]].itertuples(index=False, name=None):
        row_chars: set[str] = set()
        for value in row:
            text = nfc_strip(value)
            row_chars.update(text)
            for char in text:
                occurrence_counts[char] = occurrence_counts.get(char, 0) + 1
        for char in row_chars:
            row_counts[char] = row_counts.get(char, 0) + 1

    dict_chars = dictionary_orthography_chars(df)
    inventory = [
        CharInventory(
            char=char,
            rows=row_counts[char],
            occurrences=occurrence_counts.get(char, 0),
            in_dictionary_text=char in dict_chars,
            direct_unk=contains_unk(tokenizer, char),
            context_unk=contains_unk(tokenizer, f"a{char}a"),
        )
        for char in sorted(row_counts)
    ]
    return sorted(inventory, key=lambda item: (-int(item.is_unsafe), -item.rows, item.char))


def unique_grapheme_clusters(texts: list[str]) -> set[str]:
    clusters: set[str] = set()
    for text in texts:
        clusters.update(regex.findall(r"\X", text))
    return clusters


def _average_embedding_for_base(tokenizer: Any, embedding: Any, char: str) -> tuple[Any, str]:
    import torch

    base = base_char(char)
    ids = _ids(tokenizer, base) if base else []
    if not ids or tokenizer.unk_token_id in ids:
        return embedding.weight.mean(dim=0).detach().clone(), f"base={base!r} used=all-embedding-mean"
    return (
        embedding.weight[torch.tensor(ids, device=embedding.weight.device)].mean(dim=0).detach().clone(),
        f"base={base!r} ids={ids}",
    )


def add_legit_chars(tokenizer: Any, model: Any, chars: list[str]) -> list[str]:
    if not chars:
        return []

    input_embedding = model.get_input_embeddings()
    init_vectors = {
        char: _average_embedding_for_base(tokenizer, input_embedding, char)
        for char in chars
    }
    old_vocab_size = len(tokenizer)
    tokenizer.add_tokens(chars, special_tokens=False)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    added: list[str] = []
    import torch

    with torch.no_grad():
        input_embedding = model.get_input_embeddings()
        output_embedding = model.get_output_embeddings()
        for char in chars:
            token_id = tokenizer.convert_tokens_to_ids(char)
            if token_id == tokenizer.unk_token_id:
                raise ValueError(f"Token {char!r} still maps to unk after add_tokens")
            if token_id < old_vocab_size:
                continue

            vector, note = init_vectors[char]
            vector = vector.to(device=input_embedding.weight.device, dtype=input_embedding.weight.dtype)
            input_embedding.weight[token_id] = vector
            if output_embedding is not None:
                output_embedding.weight[token_id] = vector
            added.append(char)
            print(f"Initialized {char!r} id={token_id} from {note}")

    model.config.vocab_size = len(tokenizer)
    return added


def verify_clusters(tokenizer: Any, clusters: set[str], legit_chars: set[str]) -> dict[str, int]:
    residual_clusters = []
    residual_legit_clusters = []
    decode_mismatches = 0
    for cluster in sorted(clusters):
        ids = _ids(tokenizer, cluster)
        context_ids = _ids(tokenizer, f"a{cluster}a")
        has_unk = tokenizer.unk_token_id in ids or tokenizer.unk_token_id in context_ids
        if has_unk:
            residual_clusters.append(cluster)
            if any(char in legit_chars for char in cluster):
                residual_legit_clusters.append(cluster)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if decoded != cluster:
            decode_mismatches += 1

    print("\nGrapheme-cluster verification")
    print(f"unique_clusters: {len(clusters)}")
    print(f"residual_unk_clusters: {len(residual_clusters)}")
    print(f"residual_unk_clusters_with_legit_chars: {len(residual_legit_clusters)}")
    print(f"direct_decode_mismatches: {decode_mismatches}")
    if residual_clusters:
        print("residual_unk_cluster_sample:", " ".join(repr(cluster) for cluster in residual_clusters[:50]))
    if residual_legit_clusters:
        print(
            "residual_legit_unk_cluster_sample:",
            " ".join(repr(cluster) for cluster in residual_legit_clusters[:50]),
        )
    return {
        "unique_clusters": len(clusters),
        "residual_unk_clusters": len(residual_clusters),
        "residual_unk_clusters_with_legit_chars": len(residual_legit_clusters),
        "direct_decode_mismatches": decode_mismatches,
    }


def _format_char(char: str) -> str:
    codepoints = " ".join(f"U+{ord(ch):04X}" for ch in char)
    names = " + ".join(unicodedata.name(ch, "<unnamed>") for ch in char)
    return f"{char!r} {codepoints} {names}"


def print_inventory(inventory: list[CharInventory]) -> None:
    unsafe = [item for item in inventory if item.is_unsafe]
    legit = [item for item in unsafe if item.is_legit_orthography]
    artifacts = [item for item in unsafe if not item.is_legit_orthography]

    print("\nUNK character inventory")
    if not unsafe:
        print("(none)")
    for item in unsafe:
        kind = "legit" if item.is_legit_orthography else "artifact"
        checks = []
        if item.direct_unk:
            checks.append("direct")
        if item.context_unk:
            checks.append("context")
        print(
            f"{kind:8} rows={item.rows:6d} occurrences={item.occurrences:7d} "
            f"dict_text={item.in_dictionary_text!s:5} unk={'+'.join(checks):14} {_format_char(item.char)}"
        )

    print("\nClassification")
    print("legit_chars:", " ".join(item.char for item in legit) or "(none)")
    print("artifact_chars:", " ".join(item.char for item in artifacts) or "(none)")


def run(model_dir: Path, corpus: Path, output_dir: Path) -> dict[str, int]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    df = pd.read_parquet(corpus)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    inventory = build_char_inventory(df, tokenizer)
    print_inventory(inventory)
    legit_chars = [item.char for item in inventory if item.is_unsafe and item.is_legit_orthography]

    expected = {"ḭ", "ḛ", "ṵ", "Ḭ", "ẁ", "ȳ"}
    missing_expected = sorted(expected - set(legit_chars))
    if missing_expected:
        print("WARNING: expected legit chars not classified as legit:", " ".join(missing_expected))

    added = add_legit_chars(tokenizer, model, legit_chars)
    print(f"\nAdded {len(added)} normal tokens:", " ".join(added) or "(none)")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved extended model+tokenizer to {output_dir}")

    clusters = unique_grapheme_clusters(iter_normalized_texts(df))
    stats = verify_clusters(tokenizer, clusters, set(legit_chars))
    if stats["residual_unk_clusters_with_legit_chars"] != 0:
        raise AssertionError("Legit corpus chars still produce <unk> in grapheme-cluster verification")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.model_dir, args.corpus, args.output_dir)


if __name__ == "__main__":
    main()
