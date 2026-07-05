from mbay_nmt.tone_restore.lexicon_restore import LexiconRestorer


def test_restore_unambiguous_token_and_preserves_first_char_case():
    restorer = LexiconRestorer.from_texts(["dɔ̀ ɗèē"])

    assert restorer.restore("Dɔ ɗee.") == "Dɔ̀ ɗèē."


def test_restore_dominant_ambiguous_token():
    restorer = LexiconRestorer.from_texts(["ɓāl", "ɓāl", "ɓāl", "ɓāl", "ɓāl", "ɓal"])

    assert restorer.restore("ɓal") == "ɓāl"


def test_restore_keeps_non_dominant_ambiguous_token_unchanged():
    restorer = LexiconRestorer.from_texts(["ɓāl", "ɓāl", "ɓal", "ɓal"])

    assert restorer.restore("ɓal") == "ɓal"


def test_restore_keeps_oov_token_unchanged():
    restorer = LexiconRestorer.from_texts(["dɔ̀"])

    assert restorer.restore("ngan") == "ngan"
