import json

import pandas as pd
import pytest

from mbay_nmt.data_sources.build_v3 import (
    apply_v31_source_tags,
    apply_v32_restored_bible,
    build_v3,
    clean_bible_target,
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


def test_build_v32_restored_rows_keep_v31_bible_source_tag(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    books = ["GEN", "ROM", "EXO", "GAL", "JOS"]
    pd.DataFrame(
        [
            {
                "source_lang": "mbay",
                "target_lang": "eng",
                "source_text": f"mbay source {book}",
                "target_text": f"english target {book}",
                "domain": "bible",
                "provenance": f"bible:MlL:{book}.1.1",
                "split": "train",
            }
            for book in books
        ]
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

    assert nfc_strip("l’Éternel dit: “va” – attends…") == "l'Éternel dit: \"va\" - attends…"
