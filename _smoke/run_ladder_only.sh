#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a|clean_up|Loading weights|Fetching|^- '
for k in exaone24 qwen17; do
  echo "######## 크기사다리 $k ########"
  .venv/bin/python -m src.cache_acts $k 2>&1 | grep -vE "$FILT" || { echo "[실패]$k cache"; continue; }
  .venv/bin/python -m src.probe $k 2>&1 | grep -vE "$FILT"
  .venv/bin/python -m src.cache_lens $k 2>&1 | grep -vE "$FILT"
  .venv/bin/python -m src.seq_m2 $k 2>&1 | grep -vE "$FILT"
done
echo "######## 크기사다리 전체 완료 ########"
