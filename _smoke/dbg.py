"""nnsight 0.7 API 확정용 디버깅 (적재 1회로 여러 패턴 검증)."""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import torch
from src.model_io import load_model

lm, spec = load_model("exaone")
print("적재 완료, n_layers", spec.n_layers, flush=True)

# 1) 모듈 경로 접근
try:
    blocks = lm.transformer.h
    print("OK lm.transformer.h, len =", len(blocks), flush=True)
except Exception as e:
    print("FAIL transformer.h:", repr(e), flush=True)

# 2) 블록 output 차원 진단
try:
    with lm.trace("할아버지께서 신문을 "):
        o_tuple0 = lm.transformer.h[16].output[0].save()
        logits = lm.output.logits.save()
    ot = o_tuple0 if isinstance(o_tuple0, torch.Tensor) else getattr(o_tuple0, "value", o_tuple0)
    lt = logits if isinstance(logits, torch.Tensor) else getattr(logits, "value", logits)
    print(f"OK trace: output[0] ndim={ot.ndim} shape={tuple(ot.shape)} | logits shape={tuple(lt.shape)}", flush=True)
    tok = lm.tokenizer
    top = torch.topk(lt[0, -1].float(), 5).indices.tolist()
    print("   top5 다음토큰:", [tok.decode([i]) for i in top], flush=True)
    # 마지막 토큰 hidden 위치 확정
    last = ot[-1] if ot.ndim == 2 else ot[0, -1]
    print(f"   마지막토큰 hidden shape={tuple(last.shape)} norm={last.float().norm():.1f}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAIL trace:", repr(e), flush=True)

# 3) 전층 리스트 save (cache_acts 패턴) — output[0] 2D 가정 [seq,hidden]
try:
    with lm.trace("나는 사장님께 서류를 "):
        allh = [lm.transformer.h[L].output[0].save() for L in range(spec.n_layers)]
    arr = [(x if isinstance(x, torch.Tensor) else x.value) for x in allh]
    a0 = arr[0]
    last0 = a0[-1] if a0.ndim == 2 else a0[0, -1]
    print(f"OK 전층 리스트: {len(arr)}층, layer0 output[0] shape={tuple(a0.shape)} "
          f"마지막hidden={tuple(last0.shape)}", flush=True)
except Exception as e:
    print("FAIL 전층:", repr(e), flush=True)

print("=== DBG 완료 ===", flush=True)
