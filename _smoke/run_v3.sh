#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v3 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a|clean_up|Loading weights|Repo card|tqdm|^- '
for k in exaone llama qwen; do
  echo "######## v3 $k ########"
  .venv/bin/python -m src.cache_lens $k 2>&1 | grep -vE "$FILT" || { echo "[실패]$k lens"; continue; }
  .venv/bin/python -m src.seq_m2 $k 2>&1 | grep -vE "$FILT"
done
echo "######## v3 전체 완료 ########"
