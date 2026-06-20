#!/bin/bash
# Qwen·Llama 1·2단계 순차 실행 (24GB RAM이라 한 번에 한 모델만 적재).
# 각 모델: 다운로드 완료 대기 → 활성값 캐싱 → 프로빙 → 로짓 렌즈 → 분석/그림.
set -o pipefail
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp" || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling|double-check|pin a revision|clean_up_tokeniz|Loading weights|Fetching|^- '

run_model () {
  key="$1"
  echo "######## $key ########"
  echo "[대기] $key 다운로드 완료…"
  W=0
  while ! grep -q "$key 완료" _smoke/dl_$key.log 2>/dev/null; do
    sleep 20; W=$((W+20))
    [ $((W % 180)) -eq 0 ] && echo "  …$key 대기 ${W}s"
    [ $W -gt 7200 ] && { echo "[중단] $key 120분 초과"; return 2; }
  done
  echo "[완료] $key 다운로드"
  echo "==== $key [1/4] 활성값 캐싱 ===="
  $PY -m src.cache_acts "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key 캐싱"; return 3; }
  echo "==== $key [2/4] 프로빙 ===="
  $PY -m src.probe "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key 프로빙"; return 4; }
  $PY analysis/plot_probe.py "$key" 2>&1 | grep -E "저장"
  echo "==== $key [3/4] 로짓 렌즈 ===="
  $PY -m src.cache_lens "$key" 2>&1 | grep -vE "$FILT" || { echo "[실패] $key 렌즈"; return 5; }
  echo "==== $key [4/4] 렌즈 분석 ===="
  $PY analysis/analyze_lens.py "$key" 2>&1 | grep -vE "Glyph|findfont"
  echo "==== $key 1·2단계 완료 ===="
}

run_model qwen
run_model llama
echo "######## Qwen·Llama 1·2단계 전체 완료 ########"
