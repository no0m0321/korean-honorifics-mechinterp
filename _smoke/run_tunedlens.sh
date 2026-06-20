#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2
echo "[대기] forced_decode 완료…"
while ! grep -q "forced_decode 전체 완료" _smoke/forced.log 2>/dev/null; do sleep 30; done
for k in exaone llama qwen; do
  echo "#### tuned_lens $k ####"
  .venv/bin/python -m src.tuned_lens $k 2>&1 | grep -E "학습 위치|중간층최대|저장|Error|Traceback"
done
echo "#### tuned_lens 전체 완료 ####"
