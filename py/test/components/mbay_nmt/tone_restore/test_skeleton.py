import pytest

from mbay_nmt.tone_restore.skeleton import dict_to_bible_style, skeleton, strip_tones


def test_strip_tones_preserves_tilde_below_nasalization():
    assert strip_tones("Dèē-kɨ́-dḛ̀ḛ́ ā̰ ò̰ ṵ́ ḭ̀") == "Dee-kɨ-dḛḛ a̰ o̰ ṵ ḭ"


def test_skeleton_drops_all_combining_marks_and_lowercases():
    assert skeleton("Dèē-kɨ́-dḛ̀ḛ́ Õ Ã Ĩ ũ") == "deekɨdee o a i u"


def test_skeleton_hyphen_space_variant():
    assert skeleton("Dèē-kɨ́", hyphens="space") == "dee kɨ"


def test_skeleton_rejects_unknown_hyphen_mode():
    with pytest.raises(ValueError):
        skeleton("x", hyphens="keep")


def test_dict_to_bible_style_converts_nasal_vowels_and_strips_tones():
    assert dict_to_bible_style("Dèē-kɨ́-dḛ̀ḛ́ ā̰ ò̰ ṵ́ ḭ̀") == "Dee-kɨ-dẽẽ ã õ ũ ĩ"
