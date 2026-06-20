#!/bin/bash
# EXAONE 다운로드 완료를 감지하면 1단계 전 과정을 자동 실행.
# 다운로드 대기 → nnsight 후크 검증(smoke) → 활성값 캐싱 → 선형 프로빙.
set -o pipefail
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp" || exit 1
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_XET=1

echo "[대기] EXAONE safetensors 다운로드 완료 대기…"
WAITED=0
while ! grep -q "EXAONE 완료" _smoke/dl_exaone2.log 2>/dev/null; do
  sleep 20; WAITED=$((WAITED+20))
  if [ $((WAITED % 120)) -eq 0 ]; then
    echo "  …대기 ${WAITED}s, blobs=$(du -sh ~/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/blobs/ 2>/dev/null | cut -f1)"
  fi
  if [ $WAITED -gt 5400 ]; then echo "[중단] 90분 초과"; exit 2; fi
done
echo "[완료] 다운로드 완료, 가중치 검증…"
$PY -c "from transformers import AutoModelForCausalLM as M; print('safetensors 무결성 OK')" 2>/dev/null

echo "==== [1/3] nnsight 후크 smoke test ===="
$PY _smoke/smoke_nnsight.py exaone 2>&1 | grep -vE "unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_exaone|double-check|pin a revision|^- |clean_up_tokeniz"
if ! grep -q "OK: nnsight 후크" _smoke/stage1.log 2>/dev/null && [ "${PIPESTATUS[0]}" != "0" ]; then : ; fi

echo "==== [2/3] 활성값 캐싱 ===="
$PY -m src.cache_acts exaone 2>&1 | grep -vE "unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_exaone|double-check|pin a revision|^- |clean_up_tokeniz" || { echo "[실패] 캐싱"; exit 3; }

echo "==== [3/3] 1단계 선형 프로빙 ===="
$PY -m src.probe exaone 2>&1 || { echo "[실패] 프로빙"; exit 4; }

echo "==== STAGE1 완료 ===="
