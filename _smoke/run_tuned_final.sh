#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a|clean_up|Loading weights|Repo card|tqdm|^- '
echo "[대기] exaone24(사다리) 완료…"
while ! grep -q "exaone24 완료" _smoke/exaone24.log 2>/dev/null; do sleep 15; done
for k in exaone qwen; do
  echo "######## tuned_lens(메모리안전) $k ########"
  .venv/bin/python -m src.tuned_lens $k 2>&1 | grep -E "학습 위치|중간층최대|저장|Error|Traceback" | grep -vE "$FILT"
done
echo "######## tuned_lens 최종 완료 ########"
