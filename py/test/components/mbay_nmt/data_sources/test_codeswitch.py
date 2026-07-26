import pandas as pd

from mbay_nmt.data_sources.codeswitch import (
    build_word_maps,
    find_anchors,
    generate_codeswitch_rows,
)


def _row(source_lang, target_lang, source_text, target_text, domain, provenance="p:1", split="train"):
    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_text": source_text,
        "target_text": target_text,
        "domain": domain,
        "provenance": provenance,
        "split": split,
    }


def test_build_word_maps_cleans_short_glosses_and_skips_long_glosses():
    df = pd.DataFrame(
        [
            _row("mbay", "eng", "màn", "Rain (falling water).", "lexicon"),
            _row("mbay", "fra", "màn", "Pluie {nom}.", "lexicon"),
            _row("mbay", "eng", "kó̰", "a very long gloss", "lexicon"),
        ]
    )

    maps = build_word_maps(df)

    assert maps["eng"]["rain"][0].mbay == "màn"
    assert maps["fra"]["pluie"][0].mbay == "màn"
    assert "a very long gloss" not in maps["eng"]


def test_find_anchors_requires_whole_mbay_token_and_whole_target_phrase():
    df = pd.DataFrame(
        [
            _row("mbay", "eng", "kó̰", "fish", "lexicon"),
            _row("mbay", "eng", "kó̰ ɗèē.", "The fish came.", "example", provenance="ex:good"),
            _row("mbay", "eng", "kó̰b ɗèē.", "The fish came.", "example", provenance="ex:mbay-prefix"),
            _row("mbay", "eng", "kó̰ ɗèē.", "The fishery opened.", "example", provenance="ex:x-prefix"),
        ]
    )

    anchors = find_anchors(df, build_word_maps(df))

    assert [anchor.provenance for anchor in anchors] == ["ex:good"]


def test_generate_codeswitch_substitution_preserves_frame_except_swapped_pair():
    df = pd.DataFrame(
        [
            _row("mbay", "eng", "kó̰", "fish", "lexicon"),
            _row("mbay", "eng", "màn", "rain", "lexicon"),
            _row("mbay", "eng", "N kó̰ ɗāa.", "I saw fish.", "example", provenance="ex:1"),
        ]
    )

    out, stats = generate_codeswitch_rows(df, max_rows=10, seed=1, k=1)

    assert stats["anchors"] == 1
    assert {
        (row.source_lang, row.target_lang, row.source_text, row.target_text)
        for row in out.itertuples(index=False)
    } == {
        ("eng", "mbay", "I saw rain.", "N màn ɗāa."),
        ("mbay", "eng", "N màn ɗāa.", "I saw rain."),
    }


def test_generate_codeswitch_pos_guard_rejects_cross_category_swap():
    df = pd.DataFrame(
        [
            _row("mbay", "eng", "kó̰", "fish", "lexicon"),
            _row("mbay", "eng", "ɗèē", "to come", "lexicon"),
            _row("mbay", "eng", "N kó̰ ɗāa.", "I saw fish.", "example", provenance="ex:1"),
        ]
    )

    out, stats = generate_codeswitch_rows(df, max_rows=10, seed=1, k=1)

    assert stats["anchors"] == 1
    assert out.empty
