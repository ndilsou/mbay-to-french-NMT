"""Extend NLLB-200 tokenizer + model to support Mbay and auxiliary Sara-Bagirmi languages.

NLLB uses bare BCP-47-ish lang codes (e.g. `eng_Latn`, `fra_Latn`). New
language tokens are added via `add_special_tokens`, and model embeddings are
resized to match. New embeddings are initialized from the mean of
linguistically related donor languages already in NLLB.

Usage:
    tokenizer, model = extend_nllb_for_mbay(
        "facebook/nllb-200-distilled-600M",
        output_dir="models/nllb-600m-mbay-extended",
    )
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Mbay (ISO 639-3: myb) — our primary target.
MBAY_CODE = "myb_Latn"

# Auxiliary Sara-Bagirmi languages from Keegan's data. Used as training
# fuel only — not served to users. Codes are ad hoc (not ISO) since these
# aren't in NLLB-200; the token string just needs to be unique.
AUX_LANG_CODES: list[str] = [
    "bbt_Latn",  # Beboto
    "bjo_Latn",  # Bediondo
    "dba_Latn",  # Daba
    "gor_Latn",  # Gor
    "gly_Latn",  # Gulay
    "kba_Latn",  # Kaba
    "kbn_Latn",  # Kaba Na
    "lka_Latn",  # Laka
    "mgo_Latn",  # Mango
    "nar_Latn",  # Nar
    "ngm_Latn",  # Ngam
    "sba_Latn",  # Ngambay (ISO sba)
    "mwm_Latn",  # Sar (ISO mwm)
]

ALL_NEW_CODES: list[str] = [MBAY_CODE] + AUX_LANG_CODES

# Linguistically related donor languages in NLLB-200 for embedding init.
# Sango (sag) is the closest: Adamawa-Ubangi family, Central African Republic,
# Latin script. Bambara (bam) and Lingala (lin) are geographically proximate
# African Latin-script languages. French (fra) included because Mbay speakers
# are in a francophone context and codeswitching is common.
DONOR_CODES: list[str] = [
    "sag_Latn",  # Sango — closest
    "bam_Latn",  # Bambara
    "lin_Latn",  # Lingala
    "fon_Latn",  # Fon
    "fra_Latn",  # French
]

# Mapping from our short data-pipeline lang names to NLLB codes.
PIPELINE_TO_NLLB: dict[str, str] = {
    "mbay": MBAY_CODE,
    "beboto": "bbt_Latn",
    "bediondo": "bjo_Latn",
    "daba": "dba_Latn",
    "gor": "gor_Latn",
    "gulay": "gly_Latn",
    "kaba": "kba_Latn",
    "kabana": "kbn_Latn",
    "laka": "lka_Latn",
    "mango": "mgo_Latn",
    "nar": "nar_Latn",
    "ngam": "ngm_Latn",
    "ngambay": "sba_Latn",
    "sar": "mwm_Latn",
    "eng": "eng_Latn",
    "fra": "fra_Latn",
}


def extend_nllb_for_mbay(
    model_name: str = "facebook/nllb-200-distilled-600M",
    output_dir: str | Path | None = None,
    donor_codes: list[str] | None = None,
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Load NLLB, add new language tokens, init embeddings, optionally save."""
    donor_codes = donor_codes or DONOR_CODES
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Check which codes are truly new.
    existing = set(tokenizer.additional_special_tokens)
    new_codes = [c for c in ALL_NEW_CODES if c not in existing]

    if not new_codes:
        print("All language codes already present. No extension needed.")
        return tokenizer, model

    # Add new language tokens.
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.additional_special_tokens + new_codes}
    )

    old_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Compute donor centroid from existing embeddings.
    donor_ids = [tokenizer.convert_tokens_to_ids(c) for c in donor_codes]
    assert all(i < old_vocab_size for i in donor_ids), "Donor codes must be in original vocab"

    with torch.no_grad():
        input_embed = model.get_input_embeddings()
        donor_mean = input_embed.weight[donor_ids].mean(dim=0)

        # Init all new language token embeddings from donor centroid.
        for code in new_codes:
            new_id = tokenizer.convert_tokens_to_ids(code)
            assert new_id != tokenizer.unk_token_id, f"Token {code} not in vocab after add"
            input_embed.weight[new_id] = donor_mean

        # Mirror to output embeddings (they share weights via tie, but be explicit).
        output_embed = model.get_output_embeddings()
        if output_embed is not None:
            for code in new_codes:
                new_id = tokenizer.convert_tokens_to_ids(code)
                output_embed.weight[new_id] = donor_mean

    model.config.vocab_size = len(tokenizer)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        print(f"Saved extended model+tokenizer to {output_dir}")
        print(f"  Old vocab: {old_vocab_size}, New vocab: {len(tokenizer)}")
        print(f"  Added {len(new_codes)} language codes: {new_codes}")

    return tokenizer, model


def verify_tokenizer(tokenizer: AutoTokenizer) -> None:
    """Smoke test: all new lang codes resolve to valid IDs and round-trip."""
    for code in ALL_NEW_CODES:
        tid = tokenizer.convert_tokens_to_ids(code)
        assert tid != tokenizer.unk_token_id, f"{code} maps to UNK"

    # Test src_lang assignment.
    tokenizer.src_lang = MBAY_CODE
    enc = tokenizer("test", return_tensors="pt")
    mbay_id = tokenizer.convert_tokens_to_ids(MBAY_CODE)
    first_token = enc["input_ids"][0, 0].item()
    assert first_token == mbay_id, f"Expected prefix {mbay_id}, got {first_token}"
    print(f"All {len(ALL_NEW_CODES)} new codes verified OK.")


def test_roundtrip_mbay_strings(tokenizer: AutoTokenizer) -> dict[str, int]:
    """Encode/decode Mbay strings and count byte-fallback tokens."""
    test_strings = [
        "àbḭ̀yō̰o̰",
        "Dà-nā̰a̰ màn̄-kò̰o̰ tɨ́",
        "ɓāl ɗèē",
        "ŋgɨ̄-kɔ́ɔ́",
        "Mṵ̀ ń-tòō màjɨ̀ ngá̰y yē kòjō ndògɨ̄.",
        "āw ī-sā dálwàȳ ādɨ̄-m̄ m-ɔ́ỳ-ň lò.",
        "kîÈ rā-sí",
        "ì bà̰ý",
    ]

    tokenizer.src_lang = MBAY_CODE
    stats = {"total_tokens": 0, "byte_fallback": 0}
    for s in test_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        byte_fb = sum(1 for t in tokens if t.startswith("<0x"))
        stats["total_tokens"] += len(tokens)
        stats["byte_fallback"] += byte_fb
        roundtrip_ok = decoded.strip() == s
        print(f"  {'OK' if roundtrip_ok else 'MISMATCH'} [{byte_fb}/{len(tokens)} byte-fb] {s!r}")
        if not roundtrip_ok:
            print(f"    decoded: {decoded!r}")

    pct = stats["byte_fallback"] / max(stats["total_tokens"], 1) * 100
    print(f"\nByte-fallback rate: {stats['byte_fallback']}/{stats['total_tokens']} ({pct:.1f}%)")
    return stats


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "models/nllb-600m-mbay-extended"
    tok, model = extend_nllb_for_mbay(output_dir=out)
    verify_tokenizer(tok)
    test_roundtrip_mbay_strings(tok)
