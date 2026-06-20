#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v3 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a|clean_up|Loading weights|Repo card|tqdm|^- '
for k in exaone llama qwen; do
  echo "######## gen_power $k ########"
  .venv/bin/python -m src.gen_power $k 2>&1 | grep -E "경어 생성률|저장|Error|Traceback" | grep -vE "$FILT"
done
echo "######## gen_power 전체 완료 ########"
