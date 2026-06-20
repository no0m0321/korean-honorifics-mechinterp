#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2
for k in exaone llama qwen; do
  echo "#### steer_norm $k ####"
  .venv/bin/python -m src.steer_norm $k 2>&1 | grep -vE "unauthenticated|HF_TOKEN|Loading weights|deprecated|check_model|cache_position|modeling|double-check|pin a|clean_up"
done
echo "#### steer_norm 전체 완료 ####"
