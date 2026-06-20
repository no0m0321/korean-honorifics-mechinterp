#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling|double-check|pin a|clean_up|Loading weights|Fetching'
echo "[대기] 소형모델 다운로드 + 튜닝렌즈(GPU) 완료…"
while ! (grep -q "exaone24 완료" _smoke/dl_exaone24.log 2>/dev/null && grep -q "qwen17 완료" _smoke/dl_qwen17.log 2>/dev/null && grep -q "tuned_lens 전체 완료" _smoke/tunedlens.log 2>/dev/null); do sleep 30; done
echo "[완료] 전제 충족. EXAONE modeling 패치…"
.venv/bin/python _smoke/patch_exaone_modeling.py
for k in exaone24 qwen17; do
  echo "######## 크기사다리 $k ########"
  .venv/bin/python -m src.cache_acts $k 2>&1 | grep -vE "$FILT" || { echo "[실패] $k cache"; continue; }
  .venv/bin/python -m src.probe $k 2>&1 | grep -vE "$FILT"
  .venv/bin/python -m src.cache_lens $k 2>&1 | grep -vE "$FILT"
  .venv/bin/python -m src.seq_m2 $k 2>&1 | grep -vE "$FILT"
done
echo "######## 크기사다리 전체 완료 ########"
