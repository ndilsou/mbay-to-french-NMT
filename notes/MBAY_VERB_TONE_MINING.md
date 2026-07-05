# Grammatical Tone in Mbay Verbs — Empirical Derivation from the Keegan Corpus

Companion to `MBAY_GRAMMAR.md`. Produced 2026-07-05 by mining
`datasets/processed/mbay-corpus-v2.parquet` (5,688 unique lexicon entries,
9,748 unique example/expression sentences). Method: grep of lexicon glosses for
conjugation notes; tokenization of example sentences; affix stripping
(NFD-aware); pairing conjugated tokens with headwords via tone-stripped lookup;
per-tone-bearing-unit classification H/M/L. Scripts archived at
`py/datasets/raw/reference/mine*.py` (gitignored).

## 1. Person paradigm (CONFIRMED)

| Person | Marking | Tone effect on stem |
|---|---|---|
| 1sg | `m-` before V-stems; syllabic `m̄-` (mid) before C-stems | raising (§2) |
| 2sg | V-stems: bare raised stem (ī- optional, 17 tokens vs hundreds bare); C-stems: `ī-` | raising, identical to 1sg |
| 3sg | ∅ (citation form) | lexical |
| 1pl | `j-` before V, `kɨ̀-` before C | lexical (no raising) |
| 2pl | `ī-` + raised stem + `-n̄` | = 2sg + plural suffix |
| 3pl | stem + `-n̄` | lexical |
| logophoric 3 | `ǹ-`/`n-` (98+131 tokens) | lexical |

Lexicon statements (verbatim): `j-` "we (prefixed to verbs beginning with a
vowel; see also kɨ̀-, -jɨ̀, -jɨ́)"; `kɨ̀-` "we {prefixed to verbs beginning with a
consonant}"; `ī-` "you {subj. sing. or pl.}"; `-n̄` "subj. plural suffix
attached to verbs"; `tītɨ̄` "…can be preceded by subj. prefixes m̄, ī".

- 2sg bare V-stem: `ún bíkɨ̀ nò ādɨ̄-m̄` "Grab that pen for me"; `ā él tàa ngá̰y`
  "you talk a great deal".
- `-n̄` marks plurality of ANY plural subject, ordered after object suffixes:
  `àdɨ̄-m̄-n̄` "they gave me"; `ī-ɓáy-n̄ àĺ` "don't get up (pl.)"; `j-àw̄-n̄` "we go".
  `kɨ̀-` almost never co-occurs with `-n̄` (2/188), `j-` optionally (10/77).
- LIKELY: 1sg allomorph `ḿ-` (high) inside `ń … nò` definite relatives (8/8
  tokens): `Kɨ̀là ń ḿ-ɗāa nò` "the work that I did".

## 2. The 1/2-raising rule (CONFIRMED, quantitative)

Paired counts: 722 1sg tokens, 1,452 2nd-person tokens.

| Lexical class | 1/2 output | Evidence |
|---|---|---|
| V-initial L…M (àdɨ̄, àw̄, òō, ɔ̀r̄, à̰ӯ, ḭ̀ḭ̄, àndɨ̄) | all-Mid | n=233: àdɨ̄→m-ādɨ̄, àw̄→m-āw, ɔ̀r̄→m-ɔ̄r, à̰ӯ→m-ā̰y (38×) |
| V-initial all-L (ùn, èl, òy, ò̰o̰, ùdɨ̀, àsɨ̀, ɔ̀sɨ̀, ɔ̀dɨ̀, ùr) | all-High | n=68: ùn→m-ún, èl→m-él, ùdɨ̀→m-údɨ́, àsɨ̀→ásɨ́ |
| C-initial monosyll. L, active (sà, gà, ngà, jà, ɓà, hɔ̀, là, yà̰, ndà, ɓày) | High | ī-sá 46/46, m̄-sá 33/33, m̄-ngá 25/25, ī-hɔ́ 15/15, ī-yá̰ 13/13, m̄-já 10/10 |
| C-initial L statives/postures (ndì, tò, ɗà, gèe, ndìn) | unchanged L | m̄-gèe 90/90, ī-ndì 28/28, m̄-ɗà 18/18 |
| C-initial polysyllabic L / LM (ɓògɨ̀, nàjɨ̀; ɗèē, tèē, gèr̄, ndògɨ̄, dùbɨ̄, tàā) | unchanged | ī-ɓògɨ̀ 10/10, ī-ɗèē 53/53, m̄-tèē 18/18, m̄-tàā 14/14 |
| M, H, HM verbs (ɗāa, sō, gāng, ndīr, ndūgō; tél, dá; sáā, ngóō) | unchanged | ī-ɗāa 54/54, m̄-ndūgō 28/28, m̄-tél 12/12 |
| Irregular | àn "say" → áǹ (HL) | lexicon note + 9× m-áǹ |

3sg/3pl/1pl NEVER alter lexical tone: 3pl identity mappings dominate
(L→L 785, M→M 409, LM→LM 273, HM→HM 102 of 2,929; residue = toneless-homograph
lookup noise). 1pl: j-àw̄, j-àsɨ̀, kɨ̀-ɗāa, kɨ̀-sà — all lexical.

Future auxiliary: `m-ā` 1sg (116×), `ā` 2sg, `à` 3sg/3pl, `j-à` 1pl.
Prohibitive `òtɨ́ → ōtɨ́` (2nd pers.) shows the same raising.

## 3. Full attested paradigms (token counts)

| Gloss | 3sg | 1sg | 2sg/2 | 1pl | 3pl |
|---|---|---|---|---|---|
| give | àdɨ̄ (463) | m-ādɨ̄ (44) | ādɨ̄ (221) | — | àdɨ̄-n̄ (12) |
| go | àw̄ (141) | m-āw (92) | āw (40) | j-àw̄(-n̄) (19+4) | àw̄-n̄ (55) |
| see | òō (80) | m-ōo (42) | ōo (46) | j-òō (10) | òō-n̄ (18) |
| take | ùn (58) | m-ún (9) | ún (46) | j-ùn (1) | ùn-n̄ (5) |
| say/tell | èl (66) | m-él (9) | él (31) | j-èl (1) | èl-n̄ (10) |
| say (irreg.) | àn (55) | m-áǹ (9) | áǹ (6) | — | àn-n̄ (8) |
| die | òy (82) | m-óy (3) | óy (14) | j-òy (1) | òy-n̄ (15) |
| go down | ùdɨ̀ (65) | m-údɨ́ (6) | údɨ́ (6) | j-ùdɨ̀ (1) | ùdɨ̀-n̄ (5) |
| be able | àsɨ̀ (54) | m-ásɨ́ (4) | ásɨ́ (2) | j-àsɨ̀ (1) | — |
| drink/eat soft | à̰ӯ (13) | m-ā̰y (38) | ā̰y | j-à̰ӯ | à̰ӯ-n̄ |
| wear/eat boule | ò̰o̰ (89) | m-ó̰o̰ (6) | ó̰o̰ (9) | j-ò̰o̰ (6) | ò̰o̰-n̄ (16) |
| be (emph. cop.) | ḭ̀ḭ̄ (17) | m-ḭ̄ḭ (26) | ḭ̄ḭ (51) | j-ḭ̀ḭ̄ | ḭ̀ḭ̄-n̄ (2) |
| remove | ɔ̀r̄ (138) | m-ɔ̄r (9) | ɔ̄r (8) | — | ɔ̀r̄-n̄ (25) |
| eat (chewable) | sà (235) | m̄-sá (33) | ī-sá(-n̄) (46+2) | kɨ̀-sà(-n̄) (6+) | sà-n̄ (170) |
| do | ɗāa (396) | m̄-ɗāa (40) | ī-ɗāa(-n̄) (54) | kɨ̀-ɗāa (6) | ɗāa-n̄ (120) |
| come | ɗèē (119) | m̄-ɗèē (27) | ī-ɗèē (53) | kɨ̀-ɗèē (3) | ɗèē-n̄ (18) |
| want | gèe | m̄-gèe (90) | ī-gèe (24) | — | gèe-n̄ |
| sit/PROG | ndì (156) | m̄-ndì (22) | ī-ndì(-n̄) (28) | kɨ̀-ndì (1) | ndì-n̄ (31) |
| catch | hɔ̀ (122) | m̄-hɔ́ (12) | ī-hɔ́ (15) | — | hɔ̀-n̄ |
| find | ngà | m̄-ngá (25) | ī-ngá (13) | — | ngà-n̄ |
| get up | ɓày (95) | m̄-ɓáy (8) | ī-ɓáy(-n̄) (10) | kɨ̀-ɓày (4) | ɓày-n̄ (8) |
| return | tél (77) | m̄-tél (12) | ī-tél (7) | kɨ̀-tél (1) | tél-n̄ (6) |

## 4. Top function words / TAM particles (tone-differentiated)

| Word | n | Function |
|---|---|---|
| à | 2513 | future/habitual aux 3rd pers.; "then"; final Q-particle |
| ā | 284 | future aux 1/2sg |
| tɨ́ | 2163 | locative postposition |
| kɨ́ | 2078 | attributive/relativizer |
| ì | 1584 | copula |
| kɨ̀ | 1160 | "with; and; in order to" |
| àĺ | 774 | negator (clause-final) |
| nà̰ | 490 | "and, but; lest" |
| yé | 425 | emphatic final deictic |
| lò | 369 | genitive (→ lò-ḿ, lò-í…) |
| kɨ̄ | 308 | directional "towards" |
| dá | 307 | topic marker |
| nò | 295 | "that aforementioned" (closes ń-relatives) |
| hō | 293 | "and; also" |
| bè | 314 | "thus, just" (bèé before suffix) |
| nà | 212 | complementizer; "lest" |
| ànḛ̄ | 212 | "then, and then" |
| ń | 131+ | definite relative introducer |
| ngá̰y | ~200 | "very, a lot" |
| wà | 120 | interrogative/exclamative final |
| others | | ké "is it that", há̰, tā "now", tā̰á̰ "only/first", wòy "already", ɓétɨ́ "later/lest", kété "before", kàá "even", dò̰ó̰ "up", bèŕ "down", kɔ̄ɔ́ "away", ōtɨ́ prohibitive, ādɨ̄ hortative, kàdɨ̄ "in order to" |

Minimal-by-tone triple: `kɨ́` REL / `kɨ̀` "with" / `kɨ̄` "toward".

## 5. Pronominal suffixes — object vs oblique series (CONFIRMED)

| Person | Object (on V) | n | Oblique/poss. (on N/P) | n |
|---|---|---|---|---|
| 1sg | -m̄ (M) | 505 | -ḿ (H) | 1367 |
| 2sg | -ī (M) | 250 | -í (H) | 625 |
| 3sg | -á (H); refl./log. -ǹ (L) | 1694/669 | same | |
| 1pl | -jɨ̀ (L) | 24 | -jɨ́ (H) | 70 |
| 2pl | -sɨ̀ (L) | 8 | -sɨ́ (H) | 30 |
| 3pl | -dɨ́ (H) | 253 | -dɨ́ (H) | |
| instr. | -ň "with it" (sg subj.), -nèé (pl/after V) | 324/215 | | |

Coreference contrast: `bɔ̀ɔ̄-á mājɨ̄ tàa` "HIS (other's) father denied" vs
`èl-m̄ àn bɔ̀ɔ̄-ǹ ì mò̰y` "he said his OWN father was sick"; `Yèl̄ tèē kújɨ́-ǹ tɨ́`
"the bird flew out of its own nest" (Keegan 1998's mandatory subject
reference). Locative suffix series -á/-é/-í/-ó/-ɔ́/-ú vowel-harmonic, all High:
`kújɨ́-ú`, `bāa-á`, `súkɨ̄-ú`.

## 6. Infinitive k- preserves stem tone (CONFIRMED, 135 k+V pairs)

kàdɨ̄←àdɨ̄, kàā←àā, kèl←èl, kùn←ùn, kòjɨ̀←òjɨ̀, kà̰ӯ←à̰ӯ, kɔ̀r̄←ɔ̀r̄, kɔ̀sɨ̄←ɔ̀sɨ̄,
kò̰ō̰←ò̰ō̰, kùl̄←ùl̄, kìrè←ìrè, kɔ̀gɨ̄←ɔ̀gɨ̄, kɔ̀ӯ←ɔ̀ӯ, kòtɨ̀←òtɨ̀, kàl̄←àl̄ — no tonal
change in any explicit pair. Used after auxiliaries/preps: `gèe kàdɨ̄…`,
`ōtɨ́ kàā bāa-á`. Tone stays contrastive: kàdɨ̄ "to give" / kàdɨ̀ "sun" /
kádɨ́ "sugar cane"; kùn "taking" / kún̄ "to name"; kɔ̀r "fatigue" / kɔ̀r̄
"removing" / kɔ́r̄ "call" / kɔ̄r "field".

## 7. Rule summary with confidence

1. CONFIRMED — person marking: m-/m̄- 1sg; ī- or bare-raised 2; ∅ 3sg;
   j-/kɨ̀- 1pl; -n̄ plural subject suffix (after object suffixes); 2pl ī-…(-n̄).
2. CONFIRMED — 1/2-raising: V-initial LM→all-M; V-initial L→all-H; C-initial
   monosyllabic active L→H; statives/polysyllabic/M/H/HM invariant; àn→áǹ irregular.
3. CONFIRMED — 3sg/3pl/1pl preserve lexical tone.
4. CONFIRMED — future aux m-ā/ā/à/j-à; prohibitive òtɨ́→ōtɨ́.
5. CONFIRMED — object vs oblique suffix series differ by tone; -ǹ vs -á
   encodes subject-coreference.
6. CONFIRMED — infinitive k- preserves stem tone exactly.
7. LIKELY — ḿ- (H) 1sg allomorph in ń…nò relatives (8/8 tokens).
8. LIKELY — C-initial raising class = transitive/action monosyllables;
   posture/statives never raise (semantic characterization from ~40 types).
9. SPECULATIVE — "1/2 oblique = High, object = Mid(sg)/Low(pl)" as a
   system-wide template (each form confirmed; template is an abstraction).
