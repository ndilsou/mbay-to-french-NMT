# /// script
# requires-python = ">=3.11"
# dependencies = ["ctranslate2>=4.0", "transformers>=4.40", "sentencepiece", "fastapi", "uvicorn"]
# ///
"""Mbay translator demo — run G (CT2 int8) with dictionary hints behind a tiny FastAPI UI.

Run G (v3.5: register tags + tagged BT + dict hints) beats E/F in every direction;
X->mbay inputs get unambiguous-lexicon hints appended, mirroring training format.
"""
import json
import re
import unicodedata
from pathlib import Path

import ctranslate2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

BASE = Path.home() / "mbay/models"
CT2, HF = BASE / "rung-ct2-int8", BASE / "rung-hf"
LANGS = {"mbay": "myb_Latn", "fra": "fra_Latn", "eng": "eng_Latn"}
PUNCT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "«": '"', "»": '"'})
WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*", re.UNICODE)
HINTS = json.loads((Path.home() / "mbay/hint_lookup.json").read_text())

tok = AutoTokenizer.from_pretrained(str(HF))
tr = ctranslate2.Translator(str(CT2), device="cpu", inter_threads=1, intra_threads=6)
app = FastAPI(title="Mbay Translator")


class Req(BaseModel):
    text: str
    src: str
    tgt: str


def add_hints(text: str, src_lang: str) -> str:
    table = HINTS.get(src_lang, {})
    hints, used = [], set()
    for word in WORD_RE.findall(text.lower()):
        if len(word) <= 3:
            continue
        headword = table.get(word)
        if headword and headword not in used:
            hints.append((word, headword))
            used.add(headword)
        if len(hints) >= 2:
            break
    if hints:
        return text + " <hint> " + " ; ".join(f"{w} = {hw}" for w, hw in hints)
    return text


@app.post("/api/translate")
def translate(r: Req):
    text = unicodedata.normalize("NFC", r.text).translate(PUNCT).strip()
    if not text or r.src not in LANGS or r.tgt not in LANGS:
        return {"error": "bad input"}
    if r.tgt == "mbay" and r.src in ("fra", "eng"):
        text = add_hints(text, r.src)
    tok.src_lang = LANGS[r.src]
    src_tokens = tok.convert_ids_to_tokens(tok(text).input_ids)
    res = tr.translate_batch(
        [src_tokens], target_prefix=[[LANGS[r.tgt]]], beam_size=5, repetition_penalty=1.3, max_decoding_length=160
    )
    toks = [t for t in res[0].hypotheses[0] if t != LANGS[r.tgt]]
    out = tok.decode(tok.convert_tokens_to_ids(toks), skip_special_tokens=True)
    out = unicodedata.normalize("NFC", out).strip()
    if r.tgt != "mbay":
        # added-vocab mbay chars leak into eng/fra output glued to the prior word
        out = re.sub(r"[ḭḬḛṵẁẃȳɘ]", lambda m: " " + {"ḭ": "i", "Ḭ": "I", "ḛ": "e", "ṵ": "u", "ẁ": "w", "ẃ": "w", "ȳ": "y", "ɘ": "e"}[m.group()], out)
        out = re.sub(r"\s+", " ", out).strip()
    return {"translation": out}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traducteur Mbay</title>
<style>
  body { font: 18px/1.6 system-ui, sans-serif; background: #F4F5F2; color: #20241F; margin: 0; }
  main { max-width: 640px; margin: 0 auto; padding: 40px 20px; }
  h1 { font-family: Georgia, serif; }
  select, textarea, button { font: inherit; }
  .row { display: flex; gap: 10px; align-items: center; margin: 14px 0; }
  select { padding: 8px; border-radius: 6px; border: 1px solid #ccc; }
  textarea { width: 100%; min-height: 110px; padding: 12px; border-radius: 8px; border: 1px solid #ccc; font-family: Georgia, serif; font-size: 1.1em; }
  button { background: #2D4B8E; color: #fff; border: none; border-radius: 8px; padding: 12px 26px; cursor: pointer; }
  #out { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 60px; font-family: Georgia, serif; font-size: 1.15em; white-space: pre-wrap; }
  .muted { color: #71776C; font-size: .85em; }
</style></head><body><main>
  <h1>Traducteur Mbay</h1>
  <div class="row">
    <select id="src"><option value="mbay">Mbay</option><option value="fra">Français</option><option value="eng">English</option></select>
    →
    <select id="tgt"><option value="fra">Français</option><option value="mbay">Mbay</option><option value="eng">English</option></select>
    <button id="swap" style="background:#71776C">⇄</button>
  </div>
  <textarea id="text" placeholder="Écris ta phrase ici…"></textarea>
  <div class="row"><button id="go">Traduire</button><span class="muted" id="ms"></span></div>
  <div id="out"></div>
  <p class="muted">NLLB-1.3B fine-tuné (run G, v3.5 + indices lexicaux) · int8 CPU · les traductions vers le mbay sont encore expérimentales</p>
</main><script>
const $ = id => document.getElementById(id);
$('swap').onclick = () => { const s = $('src').value; $('src').value = $('tgt').value; $('tgt').value = s; };
$('go').onclick = async () => {
  const t0 = performance.now();
  $('out').textContent = '…';
  const r = await fetch('/api/translate', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text: $('text').value, src: $('src').value, tgt: $('tgt').value }) });
  const j = await r.json();
  $('out').textContent = j.translation ?? j.error;
  $('ms').textContent = Math.round(performance.now() - t0) + ' ms';
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8384)
