import pandas as pd

from mbay_nmt.data_sources.build_v3 import (
    apply_v31_source_tags,
    clean_bible_target,
    reassign_v31_bible_splits,
    strip_combining_marks,
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
    assert clean_bible_target(text) == "Car le Fils de l’homme est venu chercher..."


def test_clean_bible_target_removes_short_abbrev_refs_and_markers():
    assert clean_bible_target("Éz 34:2 parce qu’ils étaient...") == "parce qu’ils étaient..."
    assert clean_bible_target("# És 26:21 l’Éternel sort...") == "l’Éternel sort..."
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
