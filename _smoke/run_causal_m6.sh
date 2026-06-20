#!/bin/bash
# 인과 M6: EXAONE 완료 후 Llama·Qwen 엄밀 인과 측정(M3 직접 + M4 통제). 순차(한 모델씩 적재).
set -o pipefail
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp" || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling|double-check|pin a revision|clean_up_tokeniz|Loading weights|Fetching|^- '

echo "[대기] EXAONE causal_full 완료…"
W=0
while ! grep -q "저장: analysis/output/causal_exaone" _smoke/causal_exaone.log 2>/dev/null; do
  sleep 30; W=$((W+30)); [ $W -gt 3600 ] && { echo "[중단] EXAONE 60분 초과"; exit 2; }
done
echo "[완료] EXAONE causal_full"

for key in llama qwen; do
  echo "######## $key 엄밀 인과 ########"
  $PY -m src.causal_full "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key"; }
done
echo "######## 인과 M6 전체 완료 ########"
