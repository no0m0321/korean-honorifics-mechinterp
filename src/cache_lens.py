"""2단계 로짓 렌즈 (가설 M2): 추월층(crossover layer).

분기 직전 위치(경어형/평형형이 갈라지기 직전)에서 각 층 잔차를 최종 정규화+출력 임베딩으로
사영(로짓 렌즈)하여, 경어형 첫 분기 토큰과 평형형 첫 분기 토큰의 로짓 차를 층별로 본다.
  logit_diff_L = logit_L(경어형 첫토큰) − logit_L(평형형 첫토큰)
예측(M2): 주체 -시-(굴절)는 후기 층에서 logit_diff가 음→양으로 전환하는 '추월층'이
존재한다. 객체 보충법은 평형 어휘가 초기부터 우세하여 전 층에서 음수(추월 부재).

산출: analysis/output/lens_<model>.npz  (logit_diff[item, layer])
검증: .venv/bin/python -m src.cache_lens <model> test
실행: .venv/bin/python -m src.cache_lens <model>
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.align import align_item  # noqa: E402
from src.model_io import layer_container, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _resolve(root, dotted):
    obj = root
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _base_model(lm):
    for attr in ("_model", "model", "_module"):
        m = getattr(lm, attr, None)
        if m is not None and hasattr(m, "lm_head"):
            return m
    return lm._model


def main():
    import os
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    test = len(sys.argv) > 2 and sys.argv[2] == "test"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs = f"data/stimuli/pairs_{stim}.jsonl" if stim else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs).read_text().splitlines()]
    lm, spec = load_model(key)
    tok = lm.tokenizer
    base = _base_model(lm)
    norm_mod = _resolve(base, spec.norm_path)
    head_mod = _resolve(base, spec.head_path)
    dev = head_mod.weight.device
    dt = head_mod.weight.dtype
    bos = tok.bos_token_id
    nL = spec.n_layers
    print(f"[{key}] base={type(base).__name__} norm={type(norm_mod).__name__} "
          f"head={type(head_mod).__name__} dev={dev} dtype={dt} bos={bos}", flush=True)

    if test:
        s = [it for it in items if it["axis"] == "subject"][:4]
        o = [it for it in items if it["axis"] == "object"][:4]
        sub = s + o
    else:
        sub = items
    diff = np.full((len(sub), nL), np.nan, dtype=np.float32)
    meta = []
    for i, it in enumerate(sub):
        a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
        if a.hon_first is None or a.pln_first is None:
            meta.append(it); continue
        ids = ([bos] if bos is not None else []) + a.prefix_ids
        ids_t = torch.tensor([ids])
        saved = []
        with lm.trace(ids_t):
            blocks = layer_container(lm, spec)
            for L in range(nL):
                saved.append(blocks[L].output[0].save())
        for L in range(nL):
            h = saved[L]
            h = h if isinstance(h, torch.Tensor) else h.value
            last = h[-1] if h.ndim == 2 else h[0, -1]
            with torch.no_grad():
                lg = head_mod(norm_mod(last.to(device=dev, dtype=dt)))
            diff[i, L] = float((lg[a.hon_first] - lg[a.pln_first]).item())
        meta.append(it)
        if test:
            print(f"  [{it['axis']:7}/{it['cond']:9}] L0={diff[i,0]:+.1f} "
                  f"L{nL//2}={diff[i,nL//2]:+.1f} L{nL-1}={diff[i,nL-1]:+.1f}", flush=True)
        elif (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sub)}", flush=True)

    if test:
        print("test OK", flush=True)
        return

    axis = np.array([m["axis"] for m in meta])
    cond = np.array([m["cond"] for m in meta])
    lexeme = np.array([m["lexeme_id"] for m in meta])
    outp = ROOT / f"analysis/output/lens_{key}{suf}.npz"
    np.savez_compressed(outp, diff=diff, axis=axis, cond=cond, lexeme=lexeme)
    print(f"[{key}] 저장 {outp.relative_to(ROOT)}  diff={diff.shape}", flush=True)


if __name__ == "__main__":
    main()
