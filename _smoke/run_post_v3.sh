#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a|clean_up|Loading weights|Repo card|tqdm|^- '
echo "[대기] v3 완료…"
while ! grep -q "v3 전체 완료" _smoke/v3.log 2>/dev/null; do sleep 20; done
echo "######## 튜닝렌즈(재작성·충분학습) ########"
for k in exaone qwen; do
  echo "#### tuned_lens $k ####"
  STIM=v2 .venv/bin/python -m src.tuned_lens $k 2>&1 | grep -E "학습 위치|중간층최대|저장|Error|Traceback" | grep -vE "$FILT"
done
echo "######## 생성 검정력(v3 3범주) ########"
for k in exaone llama qwen; do
  echo "#### gen_power $k ####"
  STIM=v3 .venv/bin/python -m src.gen_power $k 2>&1 | grep -E "경어 생성률|저장|Error|Traceback" | grep -vE "$FILT"
done
echo "######## post_v3 전체 완료 ########"
