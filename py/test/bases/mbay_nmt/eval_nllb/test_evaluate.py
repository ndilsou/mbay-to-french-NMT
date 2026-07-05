from mbay_nmt.eval_nllb.evaluate import chrf_score


def test_chrf_score_strip_marks_scores_toned_and_untoned_pair_as_match():
    toned = "àbḭ̀yō̰o̰ ɓāl ɗèē ŋgɨ̄"
    untoned = "abiyoo ɓal ɗee ŋgɨ"

    assert chrf_score([toned], [untoned], strip_marks=True) == 100.0
