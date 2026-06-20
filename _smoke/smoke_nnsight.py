"""nnsight 후크 실현가능성 smoke test (연구 최대 위험 검증).

검증 항목:
  1) 모델 적재(fp16/MPS)가 24GB RAM에서 성공하는가
  2) nnsight trace로 임의 층 잔차흐름 hidden state를 캐싱할 수 있는가 (EXAONE custom code 포함)
  3) 최종 logits를 받아 경어형/평형형 표적 토큰의 로짓을 비교할 수 있는가

실행: .venv/bin/python _smoke/smoke_nnsight.py exaone
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import torch  # noqa: E402

from src.model_io import SPECS, layer_container, load_model  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "exaone"
spec = SPECS[KEY]
print(f"[1] 적재 시작: {KEY} ({spec.repo})", flush=True)
lm, spec = load_model(KEY)
print(f"    적재 완료. n_layers={spec.n_layers}", flush=True)

tok = lm.tokenizer
prompt = "할아버지께서 신문을 "
probe_layers = [0, spec.n_layers // 2, spec.n_layers - 1]
print(f"[2] trace: prompt={prompt!r}, 층={probe_layers}", flush=True)

with lm.trace(prompt):
    blocks = layer_container(lm, spec)
    saved = {L: blocks[L].output[0].save() for L in probe_layers}
    logits = lm.lm_head.output.save() if hasattr(lm, "lm_head") else lm.output.logits.save()

for L in probe_layers:
    h = saved[L]
    t = h if isinstance(h, torch.Tensor) else h.value
    print(f"    layer{L:2d} 잔차 shape={tuple(t.shape)} dtype={t.dtype} "
          f"norm={t.float().norm().item():.1f}", flush=True)

lg = logits if isinstance(logits, torch.Tensor) else logits.value
lg = lg[0, -1].float()  # 마지막 위치 다음 토큰 분포
print(f"[3] logits shape={tuple(lg.shape)}", flush=True)

# 경어형 vs 평형형 첫 토큰 로짓 비교(분기 토큰 정렬 검증의 맛보기)
for hon, pln in [("읽으셨다", "읽었다"), ("드렸다", "주었다")]:
    h_ids = tok(hon, add_special_tokens=False)["input_ids"]
    p_ids = tok(pln, add_special_tokens=False)["input_ids"]
    print(f"    {hon}={h_ids}  vs  {pln}={p_ids}", flush=True)

top5 = torch.topk(lg, 5)
print("[3] 다음 토큰 top5:", [tok.decode([i]) for i in top5.indices.tolist()], flush=True)
print("OK: nnsight 후크 실현가능성 확인", flush=True)
