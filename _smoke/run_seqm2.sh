#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2
for k in exaone llama qwen; do
  echo "#### seq_m2 $k ####"
  .venv/bin/python -m src.seq_m2 $k 2>&1 | grep -vE "unauthenticated|HF_TOKEN|Loading weights|deprecated|check_model|cache_position|modeling|double-check|pin a|clean_up"
done
echo "#### seq_m2 전체 완료 ####"
