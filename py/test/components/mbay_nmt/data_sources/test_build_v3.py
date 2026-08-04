import json

import pandas as pd
import pytest

from mbay_nmt.data_sources.build_v3 import (
    _load_extra_train_rows,
    apply_v31_source_tags,
    apply_v32_restored_bible,
    build_v3,
    clean_bible_target,
    clean_example_expression_target,
    filter_replay,
    reassign_v31_bible_splits,
    strip_combining_marks,
    verify_restored_bible_ready,
)


def test_clean_bible_target_removes_bracketed_cross_references():
    assert clean_bible_target("Il arriva [ John 19:39 ] ; puis il parla.") == "Il arriva puis il parla."


def test_clean_bible_target_removes_inline_book_references():
    text = "Mark 16:1 ; Les femmes allèrent au tombeau Ps. 23:1-3 ; avec des aromates."
    assert clean_bible_target(text) == "Les femmes allèrent au tombeau avec des aromates."


def test_clean_bible_target_strips_orphaned_punctuation_runs():
    assert clean_bible_target("; ; La parole ; ; resta.") == "La parole resta."


def test_clean_bible_target_removes_see_ch_marker_reference():
    text = "That food offered to idols is anything, or that See ch. 8:4 an idol is anything?"
    assert clean_bible_target(text) == "That food offered to idols is anything, or that an idol is anything?"


def test_clean_bible_target_removes_leading_adjacent_refs():
    text = "15:24 18:11 Car le Fils de l’homme est venu chercher..."
    assert clean_bible_target(text) == "Car le Fils de l'homme est venu chercher..."


def test_clean_bible_target_removes_short_abbrev_refs_and_markers():
    assert clean_bible_target("Éz 34:2 parce qu’ils étaient...") == "parce qu'ils étaient..."
    assert clean_bible_target("# És 26:21 l’Éternel sort...") == "l'Éternel sort..."
    assert clean_bible_target("R 11:2 et que leurs filles...") == "et que leurs filles..."
    assert clean_bible_target("voir Mt 5:3 les pauvres...") == "les pauvres..."


def test_clean_bible_target_removes_orphan_brackets():
    assert clean_bible_target("I have 17:21 ] spread you abroad") == "I have spread you abroad"


def test_clean_bible_target_removes_integer_runs_adjacent_to_refs():
    text = "among 9:30 11:1 12:17 15:1 16:2 40 21:7 17 16 the brothers Or brothers and sisters"
    assert clean_bible_target(text) == "among the brothers Or brothers and sisters"


def test_clean_bible_target_preserves_non_adjacent_standalone_numbers():
    assert clean_bible_target("10 17 the deceitfulness of riches") == "10 17 the deceitfulness of riches"
    assert clean_bible_target("les 12 apôtres sont venus") == "les 12 apôtres sont venus"


def test_build_v3_drops_bible_artifact_rows_after_cleaning(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    rows = [
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "source apparatus words here",
            "target_text": "For ver 47-49, see the note.",
            "domain": "bible",
            "provenance": "bible:MlL:GEN.1.1",
            "split": "train",
        },
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "source integer words here",
            "target_text": "13 15 17 and the words continue.",
            "domain": "bible",
            "provenance": "bible:MlL:GEN.1.2",
            "split": "train",
        },
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "source number words here",
            "target_text": "The list says: 13 15 17 beside him.",
            "domain": "bible",
            "provenance": "bible:MlL:GEN.1.3",
            "split": "train",
        },
        {
            "source_lang": "mbay",
            "target_lang": "fra",
            "source_text": "source digits words here",
            "target_text": "131, 174",
            "domain": "bible",
            "provenance": "bible:MlL:GEN.1.4",
            "split": "train",
        },
        {
            "source_lang": "mbay",
            "target_lang": "fra",
            "source_text": "source truncated",
            "target_text": "Lord forgot them",
            "domain": "bible",
            "provenance": "bible:MlL:GEN.1.5",
            "split": "train",
        },
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": "source valid number words here",
            "target_text": "He saw 10 17 men in the city.",
            "domain": "bible",
            "provenance": "bible:MlL:GEN.1.6",
            "split": "train",
        },
    ]
    pd.DataFrame(rows).to_parquet(corpus_path, index=False)

    built, counts = build_v3(corpus_path, no_notone_aug=True)

    assert counts["dropped_bible_apparatus_remnant"] == 1
    assert counts["dropped_bible_initial_integer_run"] == 2
    assert counts["dropped_bible_mostly_digits_punct"] == 1
    assert counts["dropped_bible_truncated_target"] == 1
    assert counts["bible_truncation_threshold_chars"] == 20
    assert set(built["provenance"]) == {"bible:MlL:GEN.1.6"}


def test_clean_example_expression_target_strips_trailing_gloss_suffixes():
    assert clean_example_expression_target("Je suis fatigué [Lit: je-suis.fatigué]") == "Je suis fatigué"
    assert clean_example_expression_target("prendre la route (= aller)") == "prendre la route"
    assert clean_example_expression_target("gardez [avec mon petit doigt]") == "gardez [avec mon petit doigt]"


def test_build_v3_cleans_example_expression_targets_without_dropping_rows(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    rows = [
        {
            "source_lang": "mbay",
            "target_lang": "fra",
            "source_text": "m-āw kɨ́ ɗā",
            "target_text": "Je suis fatigué [Lit: je-suis.fatigué]",
            "domain": "example",
            "provenance": "example:1",
            "split": "train",
        },
        {
            "source_lang": "mbay",
            "target_lang": "fra",
            "source_text": "kùlā ɗā",
            "target_text": "prendre la route (= aller)",
            "domain": "expression",
            "provenance": "expression:1",
            "split": "train",
        },
    ]
    pd.DataFrame(rows).to_parquet(corpus_path, index=False)

    built, counts = build_v3(corpus_path, no_notone_aug=True)

    originals = built[~built["provenance"].str.startswith("flip:")]
    assert counts["cleaned_example_expression_gloss_suffix"] == 2
    assert set(originals["target_text"]) == {"Je suis fatigué", "prendre la route"}
    assert len(originals) == 2


def test_filter_replay_drops_document_artifacts_and_weak_pairs():
    df = pd.DataFrame(
        [
            {
                "source_text": "This sentence has clean source words.",
                "target_text": "Cette phrase contient des mots propres.",
            },
            {"source_text": "Paragraph mark is here ¶ now.", "target_text": "Phrase française assez longue."},
            {"source_text": "The BAR heading appears here.", "target_text": "Phrase française assez longue."},
            {"source_text": "Read http://example.com today.", "target_text": "Phrase française assez longue."},
            {"source_text": "- I am speaking now.", "target_text": "Phrase française assez longue."},
            {
                "source_text": "This source sentence is far too long for the target side.",
                "target_text": "Trop court.",
            },
            {"source_text": "Too short.", "target_text": "Phrase française assez longue."},
        ]
    )

    filtered = filter_replay(df)

    assert filtered[["source_text", "target_text"]].to_dict("records") == [
        {
            "source_text": "This sentence has clean source words.",
            "target_text": "Cette phrase contient des mots propres.",
        }
    ]


def test_strip_combining_marks_preserves_mbay_base_letters():
    assert strip_combining_marks("àbḭ̀yō̰o̰ ɓāl ɗèē ŋgɨ̄") == "abiyoo ɓal ɗee ŋgɨ"


def test_reassign_v31_bible_splits_holds_out_whole_books():
    df = pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": f"source {book} {verse}",
                "target_text": f"target {book} {verse}",
                "domain": "bible",
                "provenance": f"bible:MlL:{book}.1.{verse}",
                "split": "train",
            }
            for book in ["GEN", "ROM", "EXO", "GAL", "JOS"]
            for verse in [1, 2]
        ]
    )

    reassigned, info = reassign_v31_bible_splits(df, val_books=("GEN", "ROM"), test_books=("EXO", "GAL"))

    assert info["val_rows"] == 4
    assert info["test_rows"] == 4
    assert set(reassigned.loc[reassigned["split"] == "val", "provenance"].str.extract(r"MlL:([^.]+)\.")[0]) == {
        "GEN",
        "ROM",
    }
    assert set(reassigned.loc[reassigned["split"] == "test", "provenance"].str.extract(r"MlL:([^.]+)\.")[0]) == {
        "EXO",
        "GAL",
    }
    train_books = set(reassigned.loc[reassigned["split"] == "train", "provenance"].str.extract(r"MlL:([^.]+)\.")[0])
    assert not train_books & {"GEN", "ROM", "EXO", "GAL"}


def test_apply_v31_source_tags_tags_bible_only_and_leaves_dictionary_flips_untagged():
    df = pd.DataFrame(
        [
            {
                "source_text": "bible source",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.1",
            },
            {
                "source_text": "dictionary source",
                "domain": "lexicon",
                "provenance": "lexicon:1",
            },
            {
                "source_text": "flipped dictionary source",
                "domain": "lexicon",
                "provenance": "flip:lexicon:1",
            },
        ]
    )

    tagged = apply_v31_source_tags(df)

    assert tagged.loc[0, "source_text"] == "<bible> bible source"
    assert tagged.loc[1, "source_text"] == "dictionary source"
    assert tagged.loc[2, "source_text"] == "flipped dictionary source"


def _restored_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "bible_text": "untoned",
        "confidence": 0.8,
        "n_chunks": 1,
        "second_text": "restored",
        "agreement": True,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_apply_v32_restored_bible_gates_thresholds_and_prefixes(tmp_path):
    corpus = pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "restored clean source only",
                "target_text": "dictionary target",
                "domain": "example",
                "provenance": "pg:examples:1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned clean",
                "target_text": "clean target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "fra",
                "source_text": "untoned source only",
                "target_text": "cible source only",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.2",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned rejected",
                "target_text": "rejected target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.3",
                "split": "train",
            },
        ]
    )
    restored_path = tmp_path / "bible-restored.parquet"
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:GEN.1.1",
                "restored_text": "restored clean",
                "roundtrip_sim": 0.95,
                "word_agreement": 0.90,
            },
            {
                "provenance": "bible:MlL:GEN.1.2",
                "restored_text": "restored source only",
                "roundtrip_sim": 0.90,
                "word_agreement": 0.89,
            },
            {
                "provenance": "bible:MlL:GEN.1.3",
                "restored_text": "restored rejected",
                "roundtrip_sim": 0.89,
                "word_agreement": 1.0,
            },
        ]
    ).to_parquet(restored_path, index=False)

    enriched, counts = apply_v32_restored_bible(corpus, restored_path)

    assert counts["v32_source_gate_verses_in"] == 2
    assert counts["v32_source_gate_verses_out"] == 1
    assert counts["v32_flip_gate_verses_in"] == 1
    assert counts["v32_flip_gate_verses_out"] == 2
    assert counts["added_v32_restored_source"] == 2
    assert counts["added_v32_restored_flip"] == 1

    restored_source = enriched[enriched["provenance"].str.startswith("restored:")]
    assert set(restored_source["source_text"]) == {"restored clean", "restored source only"}
    assert set(restored_source["target_text"]) == {"clean target", "cible source only"}

    restored_flip = enriched[enriched["provenance"].str.startswith("restored-flip:")]
    assert len(restored_flip) == 1
    assert restored_flip.iloc[0]["source_lang"] == "eng"
    assert restored_flip.iloc[0]["target_lang"] == "mbay"
    assert restored_flip.iloc[0]["source_text"] == "clean target"
    assert restored_flip.iloc[0]["target_text"] == "restored clean"
    assert restored_flip.iloc[0]["provenance"] == "restored-flip:bible:MlL:GEN.1.1"


def test_apply_v32_restored_bible_is_train_only(tmp_path):
    corpus = pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "train restored",
                "target_text": "dictionary target",
                "domain": "example",
                "provenance": "pg:examples:1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "train source",
                "target_text": "train target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "val source",
                "target_text": "val target",
                "domain": "bible",
                "provenance": "bible:MlL:ROM.1.1",
                "split": "val",
            },
        ]
    )
    restored_path = tmp_path / "bible-restored.parquet"
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:GEN.1.1",
                "restored_text": "train restored",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
            {
                "provenance": "bible:MlL:ROM.1.1",
                "restored_text": "val restored",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
        ]
    ).to_parquet(restored_path, index=False)

    enriched, counts = apply_v32_restored_bible(corpus, restored_path)

    assert counts["added_v32_restored_source"] == 1
    assert counts["added_v32_restored_flip"] == 1
    assert "val restored" not in set(enriched["source_text"])
    assert "val restored" not in set(enriched["target_text"])
    assert len(enriched[enriched["provenance"].str.contains("ROM.1.1", regex=False)]) == 1


def test_apply_v32_restored_bible_drops_non_dictionary_glyphs_but_keeps_dictionary_chars(tmp_path):
    corpus = pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "kḭ ȳ dictionary",
                "target_text": "dictionary target",
                "domain": "lexicon",
                "provenance": "pg:entries:1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned kept",
                "target_text": "kept target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned dropped",
                "target_text": "dropped target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.2",
                "split": "train",
            },
        ]
    )
    restored_path = tmp_path / "bible-restored.parquet"
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:GEN.1.1",
                "restored_text": "kḭ ȳ",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
            {
                "provenance": "bible:MlL:GEN.1.2",
                "restored_text": "Ɂér bad",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
        ]
    ).to_parquet(restored_path, index=False)

    enriched, counts = apply_v32_restored_bible(corpus, restored_path)

    assert counts["dropped_v32_restored_source_non_whitelist"] == 1
    assert counts["dropped_v32_restored_flip_non_whitelist"] == 1
    assert counts["added_v32_restored_source"] == 1
    assert counts["added_v32_restored_flip"] == 1
    assert "kḭ ȳ" in set(enriched["source_text"])
    assert "kḭ ȳ" in set(enriched["target_text"])
    assert "Ɂér bad" not in set(enriched["source_text"])
    assert "Ɂér bad" not in set(enriched["target_text"])


def test_apply_v32_restored_bible_drops_repeated_restored_candidates(tmp_path):
    corpus = pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "alpha beta gamma delta epsilon zeta eta theta dictionary",
                "target_text": "dictionary target",
                "domain": "example",
                "provenance": "pg:examples:1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned repeated gram",
                "target_text": "repeated gram target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.1",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned repeated token",
                "target_text": "repeated token target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.2",
                "split": "train",
            },
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "untoned clean restored",
                "target_text": "clean restored target",
                "domain": "bible",
                "provenance": "bible:MlL:GEN.1.3",
                "split": "train",
            },
        ]
    )
    restored_path = tmp_path / "bible-restored.parquet"
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:GEN.1.1",
                "restored_text": "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
            {
                "provenance": "bible:MlL:GEN.1.2",
                "restored_text": "alpha alpha alpha alpha beta gamma delta",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
            {
                "provenance": "bible:MlL:GEN.1.3",
                "restored_text": "alpha beta gamma delta epsilon zeta eta",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            },
        ]
    ).to_parquet(restored_path, index=False)

    enriched, counts = apply_v32_restored_bible(corpus, restored_path)

    assert counts["dropped_v32_restored_source_repetition"] == 2
    assert counts["dropped_v32_restored_flip_repetition"] == 2
    assert counts["added_v32_restored_source"] == 1
    assert counts["added_v32_restored_flip"] == 1
    assert "alpha beta gamma delta epsilon zeta eta" in set(enriched["source_text"])
    assert "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta" not in set(
        enriched["source_text"]
    )
    assert "alpha alpha alpha alpha beta gamma delta" not in set(enriched["source_text"])


def test_build_v32_restored_rows_keep_v31_bible_source_tag(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    books = ["GEN", "ROM", "EXO", "GAL", "JOS"]
    pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": "restored source GEN ROM EXO GAL JOS",
                "target_text": "dictionary target",
                "domain": "example",
                "provenance": "pg:examples:1",
                "split": "train",
            },
        ]
        + [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": f"mbay source {book}",
                "target_text": f"english target {book}.",
                "domain": "bible",
                "provenance": f"bible:MlL:{book}.1.1",
                "split": "train",
            }
            for book in books
        ],
    ).to_parquet(corpus_path, index=False)
    restored_path = tmp_path / "bible-restored.parquet"
    _restored_rows(
        [
            {
                "provenance": f"bible:MlL:{book}.1.1",
                "restored_text": f"restored source {book}",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            }
            for book in books
        ]
    ).to_parquet(restored_path, index=False)

    built, counts = build_v3(corpus_path, v32=True, restored_bible_path=restored_path)

    assert counts["added_v32_restored_source"] == 1
    assert counts["added_v32_restored_flip"] == 1
    restored_rows = built[built["provenance"].str.startswith(("restored:", "restored-flip:"))]
    assert len(restored_rows) == 2
    assert all(restored_rows["source_text"].str.startswith("<bible> "))


def test_verify_restored_bible_ready_fails_on_sha_mismatch(tmp_path):
    restored_path = tmp_path / "bible-restored.parquet"
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:GEN.1.1",
                "restored_text": "restored",
                "roundtrip_sim": 1.0,
                "word_agreement": 1.0,
            }
        ]
    ).to_parquet(restored_path, index=False)
    ready_path = tmp_path / "bible-restored.READY.json"
    ready_path.write_text(json.dumps({"status": "ready", "sha256": "not-the-real-sha"}), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_restored_bible_ready(restored_path)


def test_nfc_strip_folds_typographic_punctuation_to_tokenizer_safe_ascii():
    from mbay_nmt.data_sources.build_v3 import nfc_strip

    assert nfc_strip("l’Éternel dit: “va” – « attends »…") == "l'Éternel dit: \"va\" - \" attends \"…"


def test_load_extra_train_rows_rejects_non_train_split(tmp_path):
    extra_path = tmp_path / "codeswitch.parquet"
    pd.DataFrame(
        [
            {
                "source_lang": "eng",
                "target_lang": "mbay",
                "source_text": "fish",
                "target_text": "kó̰",
                "domain": "codeswitch",
                "provenance": "dda:ex:1",
                "split": "val",
            }
        ]
    ).to_parquet(extra_path, index=False)

    with pytest.raises(AssertionError, match="train-only"):
        _load_extra_train_rows(extra_path, label="codeswitch")


def test_build_v34_flips_full_train_lexicon_after_downsampling(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    restored_path = tmp_path / "bible-restored.parquet"
    books = ["GEN", "ROM", "EXO", "GAL", "JOS"]
    lexicon_rows = [
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": f"mbay{i}",
            "target_text": f"word{i}",
            "domain": "lexicon",
            "provenance": f"lex:{i}",
            "split": "train",
        }
        for i in range(20)
    ]
    bible_rows = [
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": f"bible source {book}",
            "target_text": f"bible target {book}.",
            "domain": "bible",
            "provenance": f"bible:MlL:{book}.1.1",
            "split": "train",
        }
        for book in books
    ]
    pd.DataFrame(lexicon_rows + bible_rows).to_parquet(corpus_path, index=False)
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:UNMATCHED.1.1",
                "restored_text": "unused",
                "roundtrip_sim": 0.0,
                "word_agreement": 0.0,
            }
        ]
    ).to_parquet(restored_path, index=False)

    built, counts = build_v3(corpus_path, v34=True, restored_bible_path=restored_path, no_notone_aug=True, seed=7)

    train_mbay_to_eng_lexicon = built[
        (built["source_lang"] == "mbay")
        & (built["target_lang"] == "eng")
        & (built["domain"] == "lexicon")
        & (built["split"] == "train")
        & ~built["provenance"].str.startswith("flip:")
    ]
    train_eng_to_mbay_lexicon = built[
        (built["source_lang"] == "eng")
        & (built["target_lang"] == "mbay")
        & (built["domain"] == "lexicon")
        & (built["split"] == "train")
        & built["provenance"].str.startswith("flip:")
    ]

    assert counts["dropped_v31_lexicon_train_downsample"] == 10
    assert counts["added_v34_full_flipped_train_lexicon"] == 20
    assert len(train_mbay_to_eng_lexicon) == 10
    assert len(train_eng_to_mbay_lexicon) == 20


def _v35_write_fixtures(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    restored_path = tmp_path / "bible-restored.parquet"
    books = ["GEN", "ROM", "EXO", "GAL", "JOS"]
    bible_rows = [
        {
            "source_lang": "mbay",
            "target_lang": "eng",
            "source_text": f"bible source {book}",
            "target_text": f"bible target {book}.",
            "domain": "bible",
            "provenance": f"bible:MlL:{book}.1.1",
            "split": "train",
        }
        for book in books
    ]
    pd.DataFrame(_v35_base_rows() + bible_rows).to_parquet(corpus_path, index=False)
    _restored_rows(
        [
            {
                "provenance": "bible:MlL:UNMATCHED.1.1",
                "restored_text": "unused",
                "roundtrip_sim": 0.0,
                "word_agreement": 0.0,
            }
        ]
    ).to_parquet(restored_path, index=False)
    return corpus_path, restored_path


def _v35_base_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(8):
        rows.append(
            {
                "source_lang": "mbay",
                "target_lang": "fra",
                "source_text": f"pàl-{i}",
                "target_text": f"faucille longue numero {i}",
                "domain": "lexicon",
                "provenance": f"lex:{i}",
                "split": "train",
            }
        )
    rows.append(
        {
            "source_lang": "mbay",
            "target_lang": "fra",
            "source_text": "Ngán-gɨ̄ ɗà tóo-n̄ àlgétɨ̀ tɨ́ nà pàl-1 tò kújɨ́ tɨ́.",
            "target_text": "Les enfants jouent avec la faucille longue dans la maison chaque matin.",
            "domain": "example",
            "provenance": "ex:1",
            "split": "train",
        }
    )
    rows.append(
        {
            "source_lang": "mbay",
            "target_lang": "fra",
            "source_text": "Dèē-gɨ̄ àw̄-n̄ ɓēe-é kɨ̀ pàl-2 lò-dɨ́ túu-bè kàrī.",
            "target_text": "Les gens rentrent au village avec leur faucille longue tous les soirs.",
            "domain": "example",
            "provenance": "ex:2",
            "split": "train",
        }
    )
    return rows


def _v35_bt_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"mbay": "Ngōn ɗà ndò̰ō̰ yá̰a̰ kɨ́ màjɨ̀.", "kind": "sentence",
             "fra": "L'enfant apprend une bonne chose utile.", "eng": "The child learns a good useful thing."},
            {"mbay": "Dèē-gɨ̄ ndì sà yá̰a̰ ɓēe-é.", "kind": "sentence",
             "fra": "Les gens mangent au village ensemble.", "eng": "too short"},
            {"mbay": "Kɨ̀là ɗāa kɨ̀ tɔ́gɨ́ ngá̰y wà.", "kind": "sentence",
             "fra": "L'enfant apprend une bonne chose utile.", "eng": "la la la la"},
        ]
    )


def test_build_v35_tags_lexicon_flips_and_reverts_upweight(tmp_path):
    corpus_path, restored_path = _v35_write_fixtures(tmp_path)

    built, counts = build_v3(corpus_path, v35=True, restored_bible_path=restored_path, no_notone_aug=True, seed=7)

    lex_flips = built[
        (built["domain"] == "lexicon") & (built["target_lang"] == "mbay")
    ]
    assert not lex_flips.empty
    assert lex_flips["source_text"].str.startswith("<lex> ").all()
    assert counts["tagged_v35_lexicon_flips"] == len(lex_flips)
    # upweight reverted: flips come from the downsampled train lexicon (50%), not the full set
    train_lex_forward = built[
        (built["domain"] == "lexicon") & (built["source_lang"] == "mbay") & (built["split"] == "train")
    ]
    assert len(lex_flips[lex_flips["split"] == "train"]) == len(train_lex_forward)
    assert counts.get("added_v34_full_flipped_train_lexicon", 0) == 0
    # forward lexicon untagged
    assert not train_lex_forward["source_text"].str.startswith("<lex>").any()


def test_build_v35_bt_rows_tagged_filtered_and_copies_added(tmp_path):
    corpus_path, restored_path = _v35_write_fixtures(tmp_path)
    bt_path = tmp_path / "bt.parquet"
    _v35_bt_frame().to_parquet(bt_path, index=False)

    built, counts = build_v3(
        corpus_path, v35=True, bt_path=bt_path, restored_bible_path=restored_path, no_notone_aug=True, seed=7
    )

    bt_rows = built[built["domain"] == "bt"]
    assert (bt_rows["split"] == "train").all()
    assert bt_rows["source_text"].str.startswith("<bt> ").all()
    assert counts["added_bt_fra"] == 2  # one dupe source dropped
    assert counts["dropped_bt_source_dupe"] == 1
    assert counts["added_bt_eng"] == 1  # "too short" and "la la la la" dropped
    assert counts["dropped_bt_source_short"] == 1
    assert counts["dropped_bt_source_degenerate"] == 1
    copies = built[built["domain"] == "copy"]
    assert counts["added_bt_copy"] == 3
    assert (copies["source_lang"] == "mbay").all()
    assert (copies["target_lang"] == "mbay").all()
    assert copies["source_text"].str.startswith("<copy> ").all()
    for _, row in copies.iterrows():
        assert row["source_text"] == "<copy> " + row["target_text"]


def test_build_v35_hints_verified_deterministic_and_train_only(tmp_path):
    corpus_path, restored_path = _v35_write_fixtures(tmp_path)
    extra = pd.read_parquet(corpus_path)
    extra = pd.concat(
        [
            extra,
            pd.DataFrame(
                [
                    {
                        "source_lang": "mbay",
                        "target_lang": "fra",
                        "source_text": "Pàl-3 tò kàdɨ̀ kújɨ́ tɨ́ nò̰o̰ kàrī bè.",
                        "target_text": "La faucille longue reste contre le mur de la maison.",
                        "domain": "example",
                        "provenance": "ex:val",
                        "split": "val",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    extra.to_parquet(corpus_path, index=False)

    built_a, counts_a = build_v3(corpus_path, v35=True, restored_bible_path=restored_path, no_notone_aug=True, seed=7)
    built_b, _ = build_v3(corpus_path, v35=True, restored_bible_path=restored_path, no_notone_aug=True, seed=7)

    pd.testing.assert_frame_equal(built_a, built_b)

    hinted = built_a[built_a["source_text"].str.contains(" <hint> ", regex=False)]
    assert (hinted["split"] == "train").all()
    assert hinted["domain"].isin({"example", "expression"}).all()
    for _, row in hinted.iterrows():
        hint_part = row["source_text"].split(" <hint> ", 1)[1]
        for clause in hint_part.split(" ; "):
            word, headword = clause.split(" = ")
            assert headword in row["target_text"]
            assert word in row["source_text"].split(" <hint> ", 1)[0].lower()
    # val/test rows never hinted
    eval_rows = built_a[built_a["split"] != "train"]
    assert not eval_rows["source_text"].str.contains("<hint>", regex=False).any()
