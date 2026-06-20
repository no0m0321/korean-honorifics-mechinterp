#!/bin/bash
# v2 정본 3모델 순차 측정(한 번에 한 모델 적재). EXAONE·Llama 준비됨, Qwen은 다운로드 대기.
set -o pipefail
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp" || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export STIM=v2
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling|double-check|pin a revision|clean_up_tokeniz|Loading weights|Fetching|^- '

runm () {
  key="$1"
  echo "######## $key (v2) ########"
  $PY -m src.cache_acts "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key cache_acts"; return 1; }
  $PY -m src.probe "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key probe"; return 1; }
  $PY analysis/plot_probe.py "$key" 2>&1 | grep -E "저장"
  $PY -m src.cache_lens "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key lens"; return 1; }
  $PY analysis/analyze_lens.py "$key" 2>&1 | grep -vE "Glyph|findfont"
  echo "######## $key (v2) 완료 ########"
}

runm exaone
runm llama
echo "[대기] qwen 다운로드 완료…"
W=0
while ! grep -q "qwen 완료" _smoke/dl_qwen.log 2>/dev/null; do
  sleep 20; W=$((W+20)); [ $((W%180)) -eq 0 ] && echo "  …qwen 대기 ${W}s"
  [ $W -gt 5400 ] && { echo "[중단] qwen 90분 초과"; break; }
done
runm qwen
echo "######## V2 3모델 전체 완료 ########"
