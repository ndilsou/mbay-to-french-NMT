# bible-restored.parquet — Handoff for corpus enrichment

Deliverable of the tone-restoration axis (2026-07-05/06). Restores Keegan
diacritics onto the 1980 ABT bible text. Background: `MBAY_GRAMMAR.md`,
`MBAY_VERB_TONE_MINING.md`.

## File

`py/datasets/processed/bible-restored.parquet` — 30,697 unique verses
(join back to corpus rows via `provenance`, `bible:MlL:<book>.<ch>.<v>`).

| column | meaning |
|---|---|
| `provenance` | verse ref, joins to corpus `provenance` |
| `bible_text` | original untoned text (as in corpus) |
| `restored_text` | byt5-base restoration, greedy, punctuation-chunked |
| `confidence` | min over chunks of exp(mean token logprob) |
| `roundtrip_sim` | folded-skeleton similarity of re-simulated bible style vs input; **catastrophe filter** (garbling/omissions/loops), blind to tone choice |
| `n_chunks` | chunks the verse was split into |
| `second_text`* | byt5-small restoration (agreement reference) |
| `agreement`* | exact match of the two models |
| `word_agreement`* | aligned word-level agreement (difflib) |

\* being recomputed after a metric fix (`e60882e`); final numbers below.

## Measured accuracy (held-out dictionary test split, full pipeline)

- byt5-base raw: **87.4%** word accuracy (greedy), 95.4% skeleton preserved
- both-models-agree subset: **91.6%** word accuracy (64.9% of short rows;
  verse-level agreement is much rarer on long real verses — use
  `word_agreement`, not `agreement`, for gating)
- residual errors: mostly tone choice on ambiguous homographs and OOV proper
  nouns — invisible to all automatic gates; needs the ~100-verse gold set for
  a true bible-domain accuracy number

## Final distributions (real bible, 30,697 verses)

Domain shift is real: cross-model word agreement median 0.60 on bible text
(vs high on dictionary-domain), concentrated in tone choices on dialect
items/OOV. Agreement is a conservative lower bound on correctness (the
reference model is the weaker one).

| gate | keeps |
|---|---|
| `roundtrip_sim ≥ 0.90` (source-grade) | 20,384 (66%) |
| `word_agreement ≥ 0.7 AND roundtrip_sim ≥ 0.95` | 6,718 (22%) |
| `word_agreement ≥ 0.8 AND roundtrip_sim ≥ 0.95` | 2,842 (9%) |
| `word_agreement ≥ 0.9 AND roundtrip_sim ≥ 0.95` | 543 (2%) |

## Recommended usage (NMT enrichment)

1. **mbay→X source side**: use `restored_text` where `roundtrip_sim ≥ 0.90`
   (66%, ~20K verses). Tone noise on the source side is benign; this is the
   safe, immediate win.
2. **X→mbay target side**: start conservative — `word_agreement ≥ 0.7 AND
   roundtrip_sim ≥ 0.95` (~6.7K verses) with per-sample weight
   `word_agreement`, alongside the (much cleaner) dictionary-domain rows.
   Escalate only after the gold-set check (below).
3. **Before scaling target-side use**: hand-check ~100 verses sampled across
   `word_agreement` bands to measure true bible-domain accuracy — the
   dictionary-domain 86-91% does NOT transfer 1:1 (this data shows it).
4. Keep the original untoned rows too (dual-register training data is free
   robustness), but never mix registers within one target sentence.
5. Proper nouns: restored forms of biblical names are model inventions
   (`Jō-hánáǹ`) — harmless for NMT but don't treat as lexicon entries.

## Known limitations

- Greedy decode (beam-4 job hit the 6h cap and was lost; +0.8pt foregone)
- True bible-domain accuracy is extrapolated from simulated-input eval;
  the input simulator was validated (char set ⊂ real bible, 68% token overlap)
  but dialect drift (e.g. `dɔning` vs `nàng`) adds unmeasured error
- `-ǹ` vs `-á` (subject-coreference) restored by model context only
- Verses with `roundtrip_sim < 0.9`: do not use restored text

## Readiness contract (for automated pickup)

Poll `py/datasets/processed/bible-restored.READY.json`:
- file absent → dataset not final, do not consume
- file present AND `"status": "ready"` → consume; all fields final

```json
{
  "status": "ready",
  "dataset": "py/datasets/processed/bible-restored.parquet",
  "s3": "s3://sagemaker-us-east-1-542301657622/mbay-tone/converted-final/<job>/output/model.tar.gz",
  "rows": 0,
  "sha256": "<parquet sha256>",
  "schema": ["provenance", "bible_text", "restored_text", "confidence",
             "roundtrip_sim", "n_chunks", "second_text", "agreement",
             "word_agreement"],
  "recommended_gates": {
    "source_side": "roundtrip_sim >= 0.90",
    "target_side": "word_agreement >= 0.70 and roundtrip_sim >= 0.95"
  },
  "docs": "notes/BIBLE_RESTORED_HANDOFF.md",
  "git_commit": "<commit of pipeline code>",
  "produced_at": "<iso8601>"
}
```

The file is written atomically (tmp + rename) strictly after the parquet and
docs are final. `rows`/`sha256` are authoritative — re-verify after copy.

## Reproduction

- Models: s3://sagemaker-us-east-1-542301657622/mbay-tone/models/{v1,v2}/,
  local `py/models/tone-restore-byt5{-base,}-sm`
- Conversion job: `mbay-tone-convert-20260705-210707` (launch:
  `python -m mbay_nmt.train_tone_restore.launch_convert`)
- Total axis cost: ~$8 SageMaker
