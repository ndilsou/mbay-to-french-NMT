# /// script
# requires-python = ">=3.11"
# dependencies = ["ctranslate2>=4.0", "transformers>=4.40", "pandas>=2.0", "pyarrow", "sentencepiece"]
# ///
"""Collapse/diversity check: translate X->mbay dict-test with a CT2 model, serving config.

Usage: uv run collapse_check.py <ct2_dir> <hf_dir> <parquet> <out_prefix>
Writes <out_prefix>.tsv (src_lang, src, hyp) and prints diversity stats.
"""
import collections
import sys
import unicodedata

import ctranslate2
import pandas as pd
from transformers import AutoTokenizer

PIPELINE_TO_NLLB = {"mbay": "myb_Latn", "eng": "eng_Latn", "fra": "fra_Latn"}
PUNCT = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"', "«": '"', "»": '"', "–": "-", "—": "-", " ": " "})

def nfc(t):
    return unicodedata.normalize("NFC", str(t)).translate(PUNCT).strip()

ct2_dir, hf_dir, parquet, out_prefix = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(hf_dir)
tr = ctranslate2.Translator(ct2_dir, device="cpu", inter_threads=2, intra_threads=4)

df = pd.read_parquet(parquet)
df = df[df["split"] == "test"] if "split" in df.columns else df
df = df[df["target_lang"] == "mbay"]

rows = []
for src_lang, g in df.groupby("source_lang"):
    if src_lang not in PIPELINE_TO_NLLB:
        continue
    tok.src_lang = PIPELINE_TO_NLLB[src_lang]
    src_tokens = [tok.convert_ids_to_tokens(tok(nfc(s)).input_ids) for s in g["source_text"]]
    prefix = [["myb_Latn"]] * len(src_tokens)
    res = tr.translate_batch(
        src_tokens, target_prefix=prefix, beam_size=5, max_batch_size=16,
        max_decoding_length=160, repetition_penalty=1.3,
    )
    for s_txt, r, ref in zip(g["source_text"], res, g["target_text"]):
        toks = [t for t in r.hypotheses[0] if t != "myb_Latn"]
        hyp = nfc(tok.decode(tok.convert_tokens_to_ids(toks), skip_special_tokens=True))
        rows.append((src_lang, nfc(s_txt), hyp, nfc(ref)))

out = pd.DataFrame(rows, columns=["src_lang", "src", "hyp", "ref"])
out.to_csv(out_prefix + ".tsv", sep="\t", index=False)

for lang, g in out.groupby("src_lang"):
    hyps = list(g["hyp"])
    uniq = len(set(hyps))
    dupes = collections.Counter(hyps).most_common(5)
    print(f"{lang}->mbay n={len(hyps)} unique={uniq} ({100*uniq/len(hyps):.0f}%)")
    for h, c in dupes:
        if c > 1:
            print(f"  x{c}: {h[:90]}")
print("saved", out_prefix + ".tsv")
