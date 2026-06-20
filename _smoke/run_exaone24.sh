#!/bin/bash
cd "/Users/swxvno/Writing a thesis 2/korean-honorifics-mechinterp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 STIM=v2 HF_HUB_OFFLINE=1
FILT='unauthenticated|HF_TOKEN|deprecated|check_model|cache_position|modeling_|double-check|pin a|clean_up|Loading weights|^- '
echo "[대기] qwen17(현 사다리) 완료…"
while ! grep -q "크기사다리 전체 완료" _smoke/ladder.log 2>/dev/null; do sleep 15; done
echo "######## exaone24 재실행 ########"
.venv/bin/python -m src.cache_acts exaone24 2>&1 | grep -vE "$FILT"
.venv/bin/python -m src.probe exaone24 2>&1 | grep -vE "$FILT"
.venv/bin/python -m src.cache_lens exaone24 2>&1 | grep -vE "$FILT"
.venv/bin/python -m src.seq_m2 exaone24 2>&1 | grep -vE "$FILT"
echo "######## exaone24 완료 ########"
