# /// script
# requires-python = ">=3.11"
# dependencies = ["ctranslate2>=4.0", "transformers>=4.40", "sacrebleu>=2.4", "pandas>=2.0", "pyarrow", "sentencepiece"]
# ///
"""Noisy-channel reranking for X->mbay: E proposes n-best, F scores round-trip.

Usage: uv run nc_rerank.py <parquet> <out_json>
Scores several lambda combos; reports chrF++ per combo per direction + collapse stats.
"""
import collections
import json
import sys
import unicodedata

import ctranslate2
import pandas as pd
import sacrebleu
from transformers import AutoTokenizer

NLLB = {"mbay": "myb_Latn", "eng": "eng_Latn", "fra": "fra_Latn"}
PUNCT = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"', "«": '"', "»": '"', "–": "-", "—": "-", " ": " "})
HOME = __import__("pathlib").Path.home()

def nfc(t):
    return unicodedata.normalize("NFC", str(t)).translate(PUNCT).strip()

parquet, out_json = sys.argv[1:3]
tok_e = AutoTokenizer.from_pretrained(str(HOME / "mbay/models/rune-hf"))
tok_f = AutoTokenizer.from_pretrained(str(HOME / "mbay/models/runf-hf"))
tr_e = ctranslate2.Translator(str(HOME / "mbay/models/rune-ct2-int8"), device="cpu", inter_threads=2, intra_threads=4)
tr_f = ctranslate2.Translator(str(HOME / "mbay/models/runf-ct2-int8"), device="cpu", inter_threads=2, intra_threads=4)

df = pd.read_parquet(parquet)
df = df[df["split"] == "test"] if "split" in df.columns else df
df = df[df["target_lang"] == "mbay"]

NBEST = 10
results = {}
for src_lang, g in df.groupby("source_lang"):
    if src_lang not in NLLB:
        continue
    srcs = [nfc(s) for s in g["source_text"]]
    refs = [nfc(r) for r in g["target_text"]]
    tok_e.src_lang = NLLB[src_lang]
    src_tokens = [tok_e.convert_ids_to_tokens(tok_e(s).input_ids) for s in srcs]
    res = tr_e.translate_batch(
        src_tokens, target_prefix=[["myb_Latn"]] * len(src_tokens),
        beam_size=NBEST, num_hypotheses=NBEST, max_batch_size=8,
        max_decoding_length=160, repetition_penalty=1.3, return_scores=True,
    )
    # collect hyps + direct scores
    all_hyps, all_direct = [], []
    for r in res:
        hyps, ds = [], []
        for toks, sc in zip(r.hypotheses, r.scores):
            toks2 = [t for t in toks if t != "myb_Latn"]
            hyps.append(nfc(tok_e.decode(tok_e.convert_tokens_to_ids(toks2), skip_special_tokens=True)))
            ds.append(sc)  # CT2 default: log prob / length (normalized)
        all_hyps.append(hyps)
        all_direct.append(ds)
    # channel scores: F scoring P(src | hyp), batched over all (sent, hyp) pairs
    tok_f.src_lang = "myb_Latn"
    flat_src, flat_tgt = [], []
    for i, hyps in enumerate(all_hyps):
        for h in hyps:
            flat_src.append(tok_f.convert_ids_to_tokens(tok_f(h if h else ".").input_ids))
            tgt_ids = tok_f(srcs[i]).input_ids
            flat_tgt.append([NLLB[src_lang]] + tok_f.convert_ids_to_tokens(tgt_ids))
    sres = tr_f.score_batch(flat_src, flat_tgt, max_batch_size=16)
    chan = []
    k = 0
    for hyps in all_hyps:
        row = []
        for _ in hyps:
            lp = sres[k].log_probs
            row.append(sum(lp) / max(len(lp), 1))
            k += 1
        chan.append(row)
    entry = {}
    for name, (a, b) in {"baseline": (1.0, 0.0), "channel_only": (0.0, 1.0), "nc_1_1": (1.0, 1.0), "nc_1_05": (1.0, 0.5), "nc_05_1": (0.5, 1.0)}.items():
        picks = []
        for i, hyps in enumerate(all_hyps):
            best = max(range(len(hyps)), key=lambda j: a * all_direct[i][j] + b * chan[i][j])
            picks.append(hyps[best])
        uniq = len(set(picks))
        entry[name] = {
            "chrf++": round(sacrebleu.corpus_chrf(picks, [refs], word_order=2).score, 2),
            "unique_pct": round(100 * uniq / len(picks)),
            "top_dupe": collections.Counter(picks).most_common(1)[0][1],
        }
        print(src_lang, name, entry[name], flush=True)
    results[src_lang] = entry

json.dump(results, open(out_json, "w"), ensure_ascii=False, indent=1)
print("saved", out_json)
