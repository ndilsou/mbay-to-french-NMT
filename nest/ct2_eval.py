# /// script
# requires-python = ">=3.11"
# dependencies = ["ctranslate2>=4.0", "transformers>=4.40", "sacrebleu>=2.4", "pandas>=2.0", "pyarrow", "sentencepiece"]
# ///
"""Evaluate a CTranslate2 int8 NLLB model per direction (forced target-lang prefix).

Usage: uv run ct2_eval.py <ct2_dir> <hf_dir_for_tokenizer> <parquet> <out_json> [beams]
"""
import json
import sys
import time
import unicodedata

import ctranslate2
import pandas as pd
import sacrebleu
from transformers import AutoTokenizer

PIPELINE_TO_NLLB = {"mbay": "myb_Latn", "eng": "eng_Latn", "fra": "fra_Latn"}

ct2_dir, hf_dir, parquet, out_json = sys.argv[1:5]
beams = int(sys.argv[5]) if len(sys.argv) > 5 else 5

tok = AutoTokenizer.from_pretrained(hf_dir)
tr = ctranslate2.Translator(ct2_dir, device="cpu", inter_threads=2, intra_threads=4)

df = pd.read_parquet(parquet)
df = df[df["split"] == "test"] if "split" in df.columns else df

PUNCT = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "})

def nfc(t):
    return unicodedata.normalize("NFC", str(t)).translate(PUNCT).strip()

def strip_marks(t):
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    )

results = {}
samples = []
t0 = time.time()
for (src, tgt), g in df.groupby(["source_lang", "target_lang"]):
    if src not in PIPELINE_TO_NLLB or tgt not in PIPELINE_TO_NLLB:
        continue
    tok.src_lang = PIPELINE_TO_NLLB[src]
    src_tokens = [tok.convert_ids_to_tokens(tok(nfc(s)).input_ids) for s in g["source_text"]]
    prefix = [[PIPELINE_TO_NLLB[tgt]]] * len(src_tokens)
    res = tr.translate_batch(
        src_tokens, target_prefix=prefix, beam_size=beams, max_batch_size=16, max_decoding_length=128
    )
    hyps = []
    for r in res:
        toks = r.hypotheses[0]
        toks = [t for t in toks if t != PIPELINE_TO_NLLB[tgt]]
        hyps.append(nfc(tok.decode(tok.convert_tokens_to_ids(toks), skip_special_tokens=True)))
    refs = [nfc(r) for r in g["target_text"]]
    key = f"{src}->{tgt}"
    entry = {
        "n": len(refs),
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
    }
    if tgt == "mbay":
        entry["chrf_stripped"] = round(
            sacrebleu.corpus_chrf([strip_marks(h) for h in hyps], [[strip_marks(r) for r in refs]], word_order=2).score, 2
        )
    results[key] = entry
    for s_txt, h, r in list(zip(g["source_text"], hyps, refs))[:8]:
        samples.append({"dir": key, "src": nfc(s_txt), "hyp": h, "ref": r})
    print(key, entry, flush=True)

results["_elapsed_s"] = round(time.time() - t0, 1)
json.dump({"metrics": results, "samples": samples}, open(out_json, "w"), ensure_ascii=False, indent=1)
print("saved", out_json)
