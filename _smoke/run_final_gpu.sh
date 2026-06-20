#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a revision|clean_up|Loading weights|Fetching|Repo card|tqdm|^- '
# Phase 1: 튜닝 렌즈 (대형 3모델)
for k in exaone llama qwen; do
  echo "######## tuned_lens $k ########"
  .venv/bin/python -m src.tuned_lens $k 2>&1 | grep -E "학습 위치|중간층최대|저장|Error|Traceback" | grep -vE "$FILT"
done
echo "######## tuned_lens 완료 ########"
# Phase 2: 크기 사다리 (소형 2모델)
echo "[대기] 소형모델 다운로드…"
W=0
while ! (grep -q "exaone24 완료" _smoke/dl_exaone24.log 2>/dev/null && grep -q "qwen17 완료" _smoke/dl_qwen17.log 2>/dev/null); do sleep 30; W=$((W+30)); [ $W -gt 3600 ] && break; done
.venv/bin/python _smoke/patch_exaone_modeling.py
for k in exaone24 qwen17; do
  echo "######## 크기사다리 $k ########"
  .venv/bin/python -m src.cache_acts $k 2>&1 | grep -vE "$FILT" || { echo "[실패]$k cache"; continue; }
  .venv/bin/python -m src.probe $k 2>&1 | grep -vE "$FILT"
  .venv/bin/python -m src.cache_lens $k 2>&1 | grep -vE "$FILT"
  .venv/bin/python -m src.seq_m2 $k 2>&1 | grep -vE "$FILT"
done
echo "######## GPU 최종 파이프라인 전체 완료 ########"
