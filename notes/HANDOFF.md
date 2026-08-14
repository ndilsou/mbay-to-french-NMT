# HANDOFF — Mbay NMT study (written 2026-08-08, migrating Mac → nest)

Read this first when resuming on nest. It replaces the Mac-local Claude memory.
Working env moves to nest (Hetzner, Helsinki, 8vCPU/32GB, tailscale). User reachable
via Claude Code remote control on nest. Repo on nest: `~/code/<this repo>`.

## TL;DR state

- **Champion: run G** (`mbay-nllb-20260804-213158`), NLLB-1.3B-v2 fine-tuned on corpus
  v3.5. Beats all prior runs in EVERY direction. Deployed on nest demo app.
- **Spend: ≈ $170 of $200 authorized** (runs A–G + ASR ≈ $148 GPU + misc). Remaining ~$30.
- **Live demo**: `nest:8384` (`~/mbay/mbay_app.py`, systemd user unit `mbay-app`,
  linger on, survives reboot). Single model rung-ct2-int8, dictionary hints on X→mbay
  inputs, repetition_penalty 1.3, added-char fold on eng/fra outputs.
- **Pending user actions**: father eval form (published artifact, French, favicon 🙏,
  https://claude.ai/code/artifact/b5bc9dbb-70d6-4718-b093-b0d4444bb52f) awaiting user
  review + WhatsApp send; results come back as text blob `Q<n>[type]=<score> | correction: ...`.
- **Demo/progress artifact** (favicon 🗣️): https://claude.ai/code/artifact/5a95a51c-c418-4c4a-81eb-93114f565878
  — still shows phase-1 numbers; update with run G when convenient.

## Scoreboard (int8 CT2, beam 5, punct-folded, fixed test sets on nest)

| chrF++ | C2 | E | F | **G** |
|---|---|---|---|---|
| bible mbay→fra | 50.9 | 52.6 | 53.58 | **53.60** |
| bible mbay→eng | 49.1 | 48.3 | 49.05 | **49.47** |
| dict mbay→fra | 20.1 | 19.8 | 19.5 | 18.8 |
| dict mbay→eng | 18.7 | 19.0 | 19.0 | 18.4 |
| dict fra→mbay | 15.3 | 14.75 | 15.34 | **17.05** (18.8 oracle-hint) |
| dict eng→mbay | 16.2 | 15.03 | 15.68 | **17.57** (20.3 oracle-hint) |
| X→mbay uniqueness (800) | — | 67–69% | 66% | **82–83%** |
| worst duplicate | — | ×36–39 | ×44 | **×21–23** |

Luna read-through verdict (all 800 rows, 2026-08-06): SHIP G — less collapsed, less
malformed (10.9% vs 20% diacritic fragments), more content-bearing (ref-token overlap
G>E on 113 pairs vs 49). G still ~1/3 template-cluster rows: direction remains hard.

## Why G won (phase-2 findings, all verified)

1. **Register bleed was the collapse root cause**: 45% of v3.4 X→mbay train targets
   were 1-word lexicon entries → model emitted headword-register templates (kɨ́-ɗāa …)
   on sentence inputs. NOT memorization (collapse strings absent from train targets).
   v3.5: `<lex> ` tag on lexicon flips + revert v3.4 ×2 upweight → 1-word share 45%→9%.
2. **~19.8K unused real toned Mbay sentences found in asr-manifest.parquet** (hidden by
   ə/ɨ orthography variant; fold ə→ɨ, drop rows w/ č Γ ° ┴ ╫ *). Back-translated with
   run F on nest (`bt_generate.py`) → 34.6K `<bt> `-tagged pairs + 19.8K `<copy> ` rows.
3. **Dictionary hints work**: train-time `src + " <hint> flûte = àlgétɨ̀"` (verified-in-
   target, p=0.5 dropout) → model follows hints at inference. Serving lookup
   (`~/mbay/hint_lookup.json`, dominant-sense ≥60% rule, ~1K words/lang) = +0.5–0.9 chrF;
   oracle gap (+2–3) remains → better lookup = future win (French lemmatization,
   sense disambiguation by context).
4. Noisy-channel rerank (E propose, F round-trip score): +1 chrF only, dupes 51→21.
   Decode-layer garnish; scripts on nest (`nc_rerank.py`). Not in serving.
5. Dead ends: silver restored-bible as target (run D, neutral-negative), DDA codeswitch
   (luna: 85% BAD), full-lexicon flip upweight (run F, made collapse worse).

## Corpus v3.5 (`datasets/processed/mbay-corpus-v3.5.parquet`, 270,005 rows)

Built by `py/components/mbay_nmt/data_sources/build_v3.py --v35` from v2 parquet +
`--replay datasets/processed/opus-replay-filtered.parquet` +
`--bt datasets/processed/bt-generated.parquet`. All v3.4 hygiene kept (bible apparatus
drops, [Lit:] strips, punct fold, anti-repetition gates, zero codeswitch assert).
S3: `s3://sagemaker-us-east-1-542301657622/mbay-nmt/data/training/mbay-corpus-v3.5.parquet`.
Tests: `py/test/components/mbay_nmt/data_sources/test_build_v3.py` (28 green).

## Infra map

- **nest `~/mbay/`**: models/{runc2,rund,rune,runf,rung}-{hf,ct2-int8}; eval/*.json,
  *-collapse.tsv, rung.log; data/test-dict.parquet (400/dir, fair all runs),
  test-bible.parquet (book holdout 1CO+1KI), test-dict-hint-{real,real2,oracle}.parquet,
  bt-candidates/bt-generated.parquet; ct2_eval.py, collapse_check.py, nc_rerank.py,
  bt_generate.py, mbay_app.py, hint_lookup.json; systemd unit
  `~/.config/systemd/user/mbay-app.service` (restart: `export XDG_RUNTIME_DIR=/run/user/$(id -u);
  systemctl --user restart mbay-app`).
- **AWS**: profile `ai-sandbox-sso` (acct 542301657622, us-east-1). MUST export
  `AWS_PROFILE=ai-sandbox-sso` for SageMaker SDK (default chain grabs stale acct
  050354499908!). Role: `arn:aws:iam::542301657622:role/service-role/AmazonSageMaker-ExecutionRole-20231002T234337`.
  SSO tokens die ~6h; jobs continue server-side; user re-runs `aws sso login`.
  NOTE: nest has NO AWS creds — training launches historically ran from the Mac.
  For future runs from nest: user must set up SSO there, or launch from wherever creds live.
- **Launcher**: from `py/`: `poe build-project sagemaker_train` then
  `uv run python -m mbay_nmt.sagemaker_train.launch --s3-data .../data/training/
  --s3-model .../model-extended-1.3b-v2/ --s3-output .../output/ --corpus-file <parquet>
  --role <arn> --instance ml.g5.2xlarge --epochs 8 --batch-size 8 --grad-accum 8
  --chrf-eval-every 2 --max-run-hours 24 --use-wandb`. NEVER pipe through `head`
  (SIGPIPE kills fit()). Run F/G config identical (~$20–23/run, ~13–15h).
- **WandB**: project mbay-NMT (creds in ~/.netrc on Mac — copy to nest if needed).
- **Polylith gotcha**: after editing components/bases, `uv sync --reinstall-package mbay-nmt`
  before running/tests.
- **codex CLI**: gpt-5.5 implementation (stalls on long tasks sometimes — verify repo
  diff appears within ~20min or take over), gpt-5.6-luna high = read-through judge
  (~45–60min for 1600 rows; output to file, never `| tail`), gpt-5.6-sol = consult
  (stalled repeatedly for us; low value, skip).
- **Eval discipline**: fixed nest test files for cross-run comparability; punct-folded
  scoring both sides; collapse/uniqueness as first-class metric; ALWAYS luna/manual
  read-through before shipping (metrics missed F's collapse regression — spotcheck
  caught it; this is standing user guidance).

## ASR / TTS axis (phase 1, unchanged)

- ASR champion: MMS-1b-all + fresh `myb_dict` adapter v1 (deeper FT regressed):
  sentence CER 16% toned / 10.9% stripped. Code `py/bases/mbay_nmt/train_asr/`.
- TTS exists: `facebook/mms-tts-myb` (verified synthesis on nest).
- Audio corpus: 13,382 clips ≈6.3h from Keegan XML ↔ S3 join; 1,570 HTML-placeholder junk.

## Next steps (ranked, ~$30 budget left)

1. **Father eval**: highest-value data for X→mbay. Form ready; parse WhatsApp replies,
   turn corrections into gold pairs (and possibly unlikelihood negatives).
2. **Run H candidate — NSL-MT unlikelihood** (~$20): negatives = G's own duplicate/
   template outputs (g-collapse.tsv clusters) + maybe French-scaffold violations;
   loss L_pos + 0.7·L_neg. Lit: +56–89% BLEU on weak directions at our scale.
   Only run if budget + evidence justify; G is a solid stopping point.
3. **Hint lookup v3** (free): French/English lemmatization (fold plurals/conjugations),
   context sense-picking. Oracle showed +2–3 chrF headroom. Prototype offline against
   test-dict-hint-oracle numbers.
4. **Dhao-style LLM post-edit** (free-ish, API pennies): retrieved dict entries +
   fuzzy-matched pairs → frontier LLM edits G's draft. +8 chrF++ in closest published
   analog. Serving latency/cost tradeoff to discuss with user.
5. Update demo artifact + study report with G numbers.

## Key literature (verified relevant)

NSL-MT (arXiv 2511.09537) unlikelihood for African LR-MT · Dhao RAG post-editing
(arXiv 2601.09982) · Tagged BT (Caswell WMT19) · Copied monolingual (Currey WMT17) ·
MAFAND (NAACL22) · Guerreiro TACL23 hallucination taxonomy · BILEX Rx (2303.15265)
lexicon augmentation · MTOB caveat: grammar books don't help, parallel examples do.

## History (compressed)

Phase 1: broken eval (missing forced_bos_token_id) diagnosed → runs A/B/C1/C2 (C2 =
1.3B v3.1 champion) → audit found U+2019→<unk> + composed-nasal→<unk> label corruption
(fixed: PUNCT fold + 8 added vocab chars ḭḛṵḬẁẃȳɘ = base model-extended-1.3b-v2) →
run D silver bible (negative) → run E v3.3 (+apostrophe fix, champion) → run F v3.4
(hygiene + flip×2; won metrics, lost luna spotcheck on X→mbay) → per-direction F/E
serving split. Phase 2 (budget→$200): register-bleed diagnosis → v3.5 → run G unifies
champion. Dates: study ran 2026-07-04 → 2026-08-06.
