# nest/ — serving + eval scripts deployed on the Hetzner box

Authoritative copies of what runs on `nest:~/mbay/`. If you edit here, scp to
`~/mbay/` and restart the app. See notes/HANDOFF.md for the full infra map.

- mbay_app.py — FastAPI demo, port 8384, run G int8 + dictionary hints (systemd
  user unit mbay-app.service, linger on)
- hint_lookup.json — dominant-sense (≥60%) fra/eng→mbay headword lookup for hints
- ct2_eval.py — per-direction chrF++/BLEU eval, punct-folded (fixed test parquets
  in nest:~/mbay/data/)
- collapse_check.py — X→mbay diversity/duplicate stats + TSV dump for read-throughs
- nc_rerank.py — noisy-channel rerank experiment (E propose, F round-trip score)
- bt_generate.py — back-translation generator (monolingual mbay → fra/eng, run F)
- rung_eval_chain.sh — convert→eval→collapse chain used for run G
