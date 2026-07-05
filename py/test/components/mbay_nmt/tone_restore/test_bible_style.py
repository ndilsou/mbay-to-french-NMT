import random

from mbay_nmt.tone_restore.bible_style import keegan_to_bible, variants


def test_strips_tones_and_maps_nasals_above():
    assert keegan_to_bible("màn̄") == "man"
    assert keegan_to_bible("dɔ̀-rā̰-gɨ̄") == "dɔrãg"


def test_drops_word_final_barred_i_after_fusion():
    assert keegan_to_bible("àdɨ̄") == "ad"
    assert keegan_to_bible("àdɨ̄-n̄") == "adɨn"


def test_collapses_long_vowels():
    assert keegan_to_bible("gòtóo") == "goto"
    assert keegan_to_bible("dèē-gɨ̄") == "deg"


def test_nasalized_y_becomes_ny():
    assert keegan_to_bible("yá̰a̰") == "nya"
    assert keegan_to_bible("yá̰a̰-gɨ̄") == "nyag"


def test_locative_postposition_cliticizes():
    assert keegan_to_bible("dɔ̀ mbàng tɨ́") == "dɔ mbangt"


def test_prefix_apostrophe():
    assert keegan_to_bible("m-ā m-āw") == "m'a m'aw"


def test_cyrillic_y_mapped():
    assert keegan_to_bible("à̰ӯ") == "ãy"


def test_full_clause():
    assert keegan_to_bible("dèē ndā jī-ǹ dɔ̀ mbàng tɨ́") == "de nda jin dɔ mbangt"


def test_variants_deterministic_given_seed():
    rng = random.Random(0)
    a = variants("bāa tɨ́", rng, n=3)
    rng = random.Random(0)
    b = variants("bāa tɨ́", rng, n=3)
    assert a == b
    assert all(v.startswith("ba") for v in a)
