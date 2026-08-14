# /// script
# requires-python = ">=3.11"
# dependencies = ["ctranslate2>=4.0", "transformers>=4.40", "pandas>=2.0", "pyarrow", "sentencepiece"]
# ///
"""Back-translate monolingual Mbay sentences to fra+eng with run F (CT2 int8).

Usage: uv run bt_generate.py <candidates_parquet> <out_parquet>
"""
import sys
import unicodedata
from pathlib import Path

import ctranslate2
import pandas as pd
from transformers import AutoTokenizer

HOME = Path.home()
tok = AutoTokenizer.from_pretrained(str(HOME / "mbay/models/runf-hf"))
tr = ctranslate2.Translator(str(HOME / "mbay/models/runf-ct2-int8"), device="cpu", inter_threads=2, intra_threads=4)

src_pq, out_pq = sys.argv[1:3]
df = pd.read_parquet(src_pq)
texts = [unicodedata.normalize("NFC", t).strip() for t in df["text"]]
tok.src_lang = "myb_Latn"
src_tokens = [tok.convert_ids_to_tokens(tok(t).input_ids) for t in texts]

out = {"mbay": texts, "kind": list(df["kind"])}
for lang_tok, col in [("fra_Latn", "fra"), ("eng_Latn", "eng")]:
    hyps = []
    B = 512
    for i in range(0, len(src_tokens), B):
        res = tr.translate_batch(
            src_tokens[i : i + B], target_prefix=[[lang_tok]] * len(src_tokens[i : i + B]),
            beam_size=2, max_batch_size=16, max_decoding_length=160,
        )
        for r in res:
            toks = [t for t in r.hypotheses[0] if t != lang_tok]
            hyps.append(unicodedata.normalize("NFC", tok.decode(tok.convert_tokens_to_ids(toks), skip_special_tokens=True)).strip())
        print(col, i + len(res), "/", len(src_tokens), flush=True)
    out[col] = hyps

pd.DataFrame(out).to_parquet(out_pq, index=False)
print("saved", out_pq)
