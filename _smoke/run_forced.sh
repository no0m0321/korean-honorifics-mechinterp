#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2
for k in exaone llama qwen; do
  echo "#### forced_decode $k ####"
  .venv/bin/python -m src.forced_decode $k 2>&1 | grep -E "baseline|저장|Error|Traceback"
done
echo "#### forced_decode 전체 완료 ####"
