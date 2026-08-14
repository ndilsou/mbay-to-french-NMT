#!/bin/bash
set -e
cd ~/mbay
echo "=== ct2 convert ==="
uvx --from ctranslate2 --with transformers,sentencepiece,torch ct2-transformers-converter \
  --model models/rung-hf --output_dir models/rung-ct2-int8 --quantization int8 --force
echo "=== eval dict ==="
uv run ct2_eval.py models/rung-ct2-int8 models/rung-hf data/test-dict.parquet eval/g-dict.json
echo "=== eval bible ==="
uv run ct2_eval.py models/rung-ct2-int8 models/rung-hf data/test-bible.parquet eval/g-bible.json
echo "=== collapse check ==="
uv run collapse_check.py models/rung-ct2-int8 models/rung-hf data/test-dict.parquet eval/g-collapse
echo ALL-DONE
