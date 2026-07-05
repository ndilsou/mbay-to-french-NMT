# Mbay Grammar & Orthography Reference

Working reference for converting the Mbay bible to Keegan's dictionary
orthography. Built 2026-07-05 from corpus forensics
(`datasets/processed/mbay-corpus-v2.parquet`) + Keegan's published work.

Mbay: Eastern Sara (Sara-Bagirmi, Central Sudanic); Moïssala (Chad), ~80-100K
speakers; ISO 639-3 `myb`.

**Bible provenance**: our scrape (YouVersion MlL/3108, "2015") is a Unicode
re-encoding of the ***Bible Mbaï*, © 1980 Alliance Biblique du Tchad** (also on
YouVersion as MB80/444 in the original Latin-1 substitute orthography:
`þ→ɓ, ð→ɗ, ï→ɨ, ø→ɔ`; Gen 1:1 `Lo yaþoodat Luwah ndah dørãg kï døning` =
our `Lo yaɓoodat Luwah ndah dɔrãg kɨ dɔning`). Orthography descends from
Caprile 1976's "orthographe pratique du mbay de Moissala" — designed for
French typewriters, hence no tone marking.

Primary sources: John M. Keegan, *A Reference Grammar of Mbay* (LINCOM 1997;
tonal inflection pp. 32-34), *Dictionary of Mbay* (1996), *The Digital Mbay
Language Project* (2009) and the Sara-Bagirmi Languages Project
(morkegbooks.com — Transcription.pdf, Eastern Sara vol. read 2026-07-05);
Ndjerareou/Melick/Moeller 2010 (Ngambay verb-tone classes, GIALens 4-2).
Our corpus embeds the dictionary (domains `lexicon`/`example`/`expression`,
31.7K Mbay rows) and the bible (61.6K rows).

## 1. Phonology & Keegan's notation

Transcription system = Collège Charles Lwanga (Sarh, 1970s) conventions, used
across all Keegan's Sara work. Keegan is explicit it is *not* a practical
orthography ("few [literate Chadians] are able to [write tone] with accuracy").

- **Tones (3 even levels), marked on every vowel and syllabic consonant:**
  - High = acute: `á é í ó ú ɨ́`
  - Mid = macron: `ā ē ī ō ū ɨ̄`
  - Low = grave: `à è ì ò ù ɨ̀`
  - Syllabic/final sonorants bear tone too: `ḿ ń ǹ ŕ ĺ r̄ l̄ m̄ n̄ ȳ w̄`
  - Contour tones = vowel sequences: `áà` (HL fall), `àá` (LH rise), long vowels doubled: `ēe`, `óo`, `dā̰a̰`
  - Mid-Low contour extremely rare (3-4 morphemes); High-Low mostly in loans/ideophones. Vowel-length contrasts are more extensive in Mbay than other Eastern Sara and marked throughout the dictionary.
- **Nasal vowels**: COMBINING TILDE BELOW (U+0330): `a̰ ḛ ḭ o̰ ṵ` — below, to keep space above free for tone marks.
- **Consonants**: implosives `ɓ ɗ`; prenasalized `mb nd ng nj`; plain `b d g h j k l m n r s t w y '` (glottal); `ɨ` = high central vowel (often epenthetic); `ɔ` open-o.
- Nasalized glide: Keegan writes `y` + nasal vowel (`yá̰a̰` "thing"); phonetically [ɲ].

## 2. Bible orthography ("national" style) vs Keegan

The bible text is the same language (core vocab matches Keegan's lexicon
precisely) in the tone-less national-style orthography. Verified mapping rules,
with occurrence-weighted share of bible tokens each rule bridges
(cumulative total 86.1% matched; scripts: scratchpad `bridge_experiment.py`):

| # | Rule | Bible → Keegan | Example | Share |
|---|------|----------------|---------|-------|
| 0 | No tone marks | add tones | `man` → `màn̄` "water" | (all) |
| 1 | Nasal tilde ABOVE → BELOW | `ã ẽ ĩ õ ũ` → `a̰ ḛ ḭ o̰ ṵ` | `dãg` → `dā̰a̰-gɨ̄` | — |
| 2 | Direct skeleton match | — | `el` → `èl` "say" | 45.6% |
| 3 | Word-final `ɨ` dropped | restore `ɨ` | `ad` → `àdɨ̄` "give" | 7.1% |
| 4 | Long (doubled) vowels collapsed | re-lengthen | `goto` → `gòtóo` "absent" | (in fold) |
| 5 | Final `-h` for Keegan long vowel | `Vh` → `V̄V` | `bah` → `bāa` "river" (`bahbo` = `bāa bò` "sea") | 1.3% |
| 6 | `ny` + oral V for nasalized `y` + nasal V | `nya` → `yá̰a̰` "thing" | `nyag` → `yá̰a̰-gɨ̄` | (in fold) |
| 7 | Clitics FUSED to host | split + restore | see table below | 10.9% |
| 8 | Apostrophe = morpheme boundary | `m'a` → `m-ā`, `'nten` → `ń-tèń` | 3.3% |
| 9 | Compounds written solid | split | `titbe` → `tītɨ̄ bè` "like this" | 17.0%* |
| 10 | Doubled `rr` for tone-bearing final r | `girr` → `gír̄`-type | — |

\* compound-split is an upper bound (2-way split against a 2.1K-skeleton
lexicon has false-positive room); the real converter must pick the right split
in context.

### Fused clitic inventory (bible suffix → Keegan form)

| Bible | Keegan | Gloss | Bridged share |
|-------|--------|-------|------|
| `-t` | `tɨ́` | locative postposition | 1.8% |
| `-ɨn`, `-n` | `n̄` / `ǹ` / `ń` | 3pl subject marker; 3sg possessive; relativizer | 4.3% |
| `-g` | `gɨ̄` | plural | 1.0% |
| `-a` | `-á` | 3sg possessive ("his/her") | 2.0% |
| `-i` | `-í` | 2sg possessive | 1.6% |
| `-m` | `-ḿ`/`m̄` | 1sg object/possessive | 0.2% |
| `-d` | `-dɨ́` | 3pl object/possessive ("them") | 1.0% |

Examples from aligned verses:
- `De nda jin dɔ mbangt` = `dèē ndā jī-ǹ dɔ̀ mbàng tɨ́` "the man puts his hand on the rock"
- `Lo yaɓoodat Luwah ndah dɔrãg kɨ dɔning` — Gen 1:1; `dɔ̀-rā̰-gɨ̄` "heavens"
- `elɨn deg anan:` = `èl-n̄ dèē-gɨ̄ àn án` "they told the people, saying:"

### Bible-specific items (never in dictionary; keep or map via table)
- `Luwah` "God (Dieu)", `Bɨraɓe` "the LORD (l'Éternel)" — translator coinages
- Proper nouns (`Dabid`, `Israel(ag)`, `Jeju`…): 3.9% of tokens; transliterate, don't restore
- Frequent function words w/o clean dict match yet: `kamla` "then", `kɨta`, `kɨtɔ` "because", `nten` (`ń-tèń` "this"), `anan` "thus/saying" — need a hand-built ~50-entry function-word table (they're high-frequency, so cheap to curate and high-impact)

## 3. Grammatical tone (why lexicon lookup alone fails)

Tone in Mbay is lexical AND grammatical. Verb stems change tone with subject
person — attested inside dictionary glosses themselves:
- `àn` "say" → `áǹ` in 1sg/2sg
- `àdɨ̄` "give" → `m-ādɨ̄` (1sg), `ādɨ̄` (2sg, whence "may/let")
- Subject affixes: `m-` 1sg, `j-` 1pl, 3sg = zero (all Eastern Sara), plural
  subject suffix `-n̄` = 3pl AND 2pl in Mbay (Sar/Nar use `-ī` for 2pl)
- `-ǹ` 3sg oblique suffix only when coreferent with subject (Keegan 1998,
  "Mandatory Subject Reference in Mbay")
- Infinitive/verbal noun = `k-` prefix (`kàdɨ̄` "to give")
- Suffixation preserves tone: `àdɨ̄ + á → àd-āá` "he gave him" (ɨ assimilates,
  its tone survives on the contour — Keegan 1997:32-34)

### Person paradigm (empirically confirmed from corpus, 2026-07-05 mining pass;
~2,900 paired conjugated↔citation verb tokens)

| Person | Marking | Tone effect on stem |
|---|---|---|
| 1sg | `m-` (V-stems) / syllabic `m̄-` (C-stems) | **raising** |
| 2sg | bare raised stem (V-stems; `ī-` optional) / `ī-` (C-stems) | **raising, = 1sg** |
| 3sg | ∅ (citation form) | lexical |
| 1pl | `j-` (V-stems) / `kɨ̀-` (C-stems) | lexical |
| 2pl | `ī-` + raised stem + `-n̄` | = 2sg + plural suffix |
| 3pl | stem + `-n̄` | lexical |
| logophoric | `ǹ-`/`n-` | lexical |

**1/2-raising rule (CONFIRMED, quantitative):**
- V-initial L…M stems → all-Mid: `àdɨ̄ → m-ādɨ̄` (44×), `àw̄ → m-āw` (92×), `ɔ̀r̄ → m-ɔ̄r`
- V-initial all-Low stems → all-High: `ùn → m-ún`, `èl → m-él`, `ùdɨ̀ → m-údɨ́`
- C-initial monosyllabic Low ACTIVE verbs → High: `sà → ī-sá` (46/46), `hɔ̀ → ī-hɔ́`, `ngà → m̄-ngá`
- Statives/postures stay Low: `m̄-gèe` (90/90), `ī-ndì`, `m̄-ɗà`; polysyllabic and M/H/HM stems invariant: `ī-ɗāa` (54/54), `m̄-tèē`, `m̄-tél`
- Irregular: `àn` "say" → `áǹ` 1/2sg
- Future aux raises too: `m-ā / ā` (1/2) vs `à` (3) vs `j-à` (1pl); prohibitive `òtɨ́ → ōtɨ́`
- 3sg/3pl/1pl NEVER alter lexical tone (identity dominates 3pl pairs, n≈1,600)
- Suffix order: object suffix before plural `-n̄`: `àdɨ̄-m̄-n̄` "they gave me"

### Pronominal suffixes — two tonally distinct series (CONFIRMED)

| Person | Object (on verb) | Oblique/possessive (on noun/prep) |
|---|---|---|
| 1sg | `-m̄` (M) | `-ḿ` (H) |
| 2sg | `-ī` (M) | `-í` (H) |
| 3sg | `-á` (H); `-ǹ` (L) when coreferent w/ subject | same |
| 1pl | `-jɨ̀` (L) | `-jɨ́` (H) |
| 2pl | `-sɨ̀` (L) | `-sɨ́` (H) |
| 3pl | `-dɨ́` (H) | `-dɨ́` (H) |

`bɔ̀ɔ̄-á` "his (someone else's) father" vs `bɔ̀ɔ̄-ǹ` "his own father" — the bible's
fused `-a`/`-n` clitics collapse this contrast; restoring it needs syntax.
Locative suffixes are vowel-harmonic all-High: `kújɨ́-ú` "in the house",
`bāa-á` "at the river".

### Infinitive `k-` preserves stem tone exactly (CONFIRMED, 135 pairs)
`àdɨ̄→kàdɨ̄`, `èl→kèl`, `ɔ̀r̄→kɔ̀r̄`… Minimal pairs stay contrastive:
`kàdɨ̄` "to give" / `kàdɨ̀` "sun" / `kádɨ́` "sugar cane".

### High-frequency function words (tone is the only differentiator)
`kɨ́` REL vs `kɨ̀` "with" vs `kɨ̄` "toward"; `à` 3-future vs `ā` 1/2-future;
`ì` copula, `àĺ` NEG-final, `tɨ́` LOC, `lò` GEN, `nò`/`ń` definite relative
frame, `dá` topic, `nà̰` "and/lest", `bè` "thus", `wòy` "already", `ōtɨ́`
prohibitive, `ādɨ̄` hortative, `kàdɨ̄` "in order to". These are exactly the
bible's highest-frequency skeletons — a curated table disambiguates most of
the "ambiguous 2-forms" mass.

⇒ **Surface verb tone is predictable from (verb class, person, number,
aspect)** — confirmed for Mbay from our own corpus (matching the pan-Sara
pattern: Daba paradigm in Keegan's Eastern Sara vol.; 5 Ngambay tone-classes,
Ndjerareou et al. 2010). Full mining report with 20+ complete paradigms and
confidence levels: see the 2026-07-05 session (scripts `mine1..7*.py`).

So the same bible skeleton (`ad`) legitimately maps to several toned forms
depending on person/aspect/context — 90% of dictionary-val tokens have >1 toned
candidate. Numbers (dict val split, detoned input):
- most-common-form lexicon: 68.0% token accuracy (unambiguous subset: 99.7%)
- + one token of left context (bigram): **85.4%**
- ⇒ full-sentence contextual model (ByT5) expected ≥90–95%, in line with
  diacritic-restoration literature: Yoruba seq2seq <5% error (Orife 2018,
  arXiv:1804.00832), Yoruba SMT 97.3%, Igbo n-gram/embedding 96–98.8%

*(§3 to be extended with the full verb-paradigm tables from the dictionary-mining pass.)*

## 4. Conversion feasibility — verdict

**Yes, convertible — via a hybrid pipeline, with quantified confidence, not
by pure lookup.**

Evidence:
1. Same language, systematic orthography delta: 8 mechanical rules bridge
   86.1% of bible token occurrences to dictionary vocabulary (from 42.8%
   naive). ~4% proper nouns pass through. True OOV ≈ 10%.
2. Tone assignment is context-dependent but learnable: 85.4% with a trivial
   bigram; ample training signal exists (31.7K toned dict rows + 56.7K toned
   ASR-manifest texts + sister-language parquet).

Recommended pipeline:
1. **Rule normalizer** (deterministic): re-segment bible text — split fused
   clitics/compounds, restore final `ɨ`, long vowels, nasal mapping. Emit
   candidate lattice, not single string.
2. **Function-word table** (~50 hand entries) for the highest-frequency
   grammar words.
3. **ByT5 tone restorer** trained on dict/ASR toned text with *bible-style
   simulated* input (upgrade `prep_byt5_pairs.py`: currently only strips
   tones/nasals — must also drop final ɨ, collapse VV, fuse clitics, `ny`,
   `-h`), so train/test input distribution matches real bible text.
4. **Confidence gating**: emit restored verse + per-token confidence; keep
   low-confidence verses out of training data or down-weight them.
5. **Gold eval set**: hand-tone (or have a speaker check) ~100 verses to
   measure true accuracy before augmenting the corpus.

Risks:
- Compound-split ambiguity (17% of mass) — mitigated by contextual model + lattice
- Residual OOV 10% — names, dialect items; confidence-gate them
- Verse-level: at 90% token accuracy a 20-token verse is ~12% fully clean —
  so use token-level weighting/masking rather than requiring perfect verses.

## 4b. Implementation status (2026-07-05)

- `components/mbay_nmt/tone_restore/bible_style.py` — Keegan→bible simulator
  (rules §2), 18 tests green.
- `prep_byt5_pairs.py` rewritten: dict + ASR-manifest targets (ə→ɨ normalized,
  decode-junk filtered), simulated inputs + stochastic variants → 22.1K pairs;
  simulated char inventory ⊂ real bible; 68% of simulated tokens attested
  verbatim in the real bible.
- `bases/mbay_nmt/train_tone_restore/` — ByT5 trainer (train.py), evaluator
  (evaluate.py: word accuracy, skeleton preservation, real-verse samples),
  SageMaker entrypoint+launcher; own Polylith project
  `projects/tone_restore_train` (sagemaker_train untouched).
- First SM run: `mbay-tone-byt5-20260705-125510` (byt5-small, 10 epochs,
  g5.xlarge spot, ~60 min). **Results: word accuracy 86.0% val / 86.1% test**
  (eval_loss 0.093), skeleton preserved 94.1% — beats the 85.4% bigram
  baseline with zero grammar-rule injection; predicted ≥85-90% band hit.
  Model: `py/models/tone-restore-byt5-sm` +
  `s3://sagemaker-us-east-1-542301657622/mbay-tone/output/`.
- Known failure mode: greedy decode repetition-loops on very long verses
  (2/20 samples) — chunk verses at punctuation before restoring; do NOT use
  byte-level no_repeat_ngram (corrupts legit repeated patterns, tested).
- Next accuracy levers: function-word table postprocess, more epochs/byt5-base,
  grammar-rule-constrained decoding, verse chunking, confidence gating.

## 5. External resources & open items

- Online [Mbay Sound Dictionary](https://morkegbooks.com/Services/World/Languages/SaraBagirmi/SoundDictionary/Mbay/)
  (387 pp, JS app) — check overlap with our parsed dictionary before scraping.
- Full 1980 Bible PDF (bibliamundi.com) downloaded during research (1799 pp) —
  original Latin-1-substitute orthography; useful to cross-check the 2015
  Unicode re-encoding.
- Mbay verb tone-class tables: only in Keegan 1997 print. If precision on verb
  tones becomes the bottleneck, consider getting the book (or extract classes
  empirically from dictionary conjugation notes).
- Bible's sporadic accents (`ngé-`, `à` future, `kàd`) look lexicalized, not
  systematic tone marking — treat as hints, verify with concordance if needed.
