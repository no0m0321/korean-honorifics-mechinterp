"""자극 잔차흐름 활성값 캐싱 (1단계 프로빙 입력).

각 자극의 프롬프트를 forward하고, **동사 생성 직전(마지막 토큰) 위치**의 모든 층 잔차
흐름 hidden state를 저장한다. M1은 '입력 문맥이 경어 요구를 인코딩하는가'를 보므로 동사
정보가 누출되지 않는 이 위치가 순수 인코딩 측정점이다.

산출: analysis/output/acts_<model>.npz
  X[n_items, n_layers, hidden], y(cond 1/0), lexeme(어휘 id), axis(subject/object),
  cond, frame, object_case
실행: .venv/bin/python -m src.cache_acts <model_key>
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

from src.model_io import layer_container, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    import os
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs = f"data/stimuli/pairs_{stim}.jsonl" if stim else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs).read_text().splitlines()]
    print(f"[{key}] 자극 {len(items)}개 적재, 모델 로딩…", flush=True)
    lm, spec = load_model(key)
    tok = lm.tokenizer
    nL = spec.n_layers
    X = None  # hidden 크기는 첫 trace에서 동적 결정(소형 모델 호환)
    for i, it in enumerate(items):
        saved = []  # nnsight 0.7: trace 내 comprehension은 밖으로 안 나옴 → 외부 append
        with lm.trace(it["prompt"]):
            blocks = layer_container(lm, spec)
            for L in range(nL):
                saved.append(blocks[L].output[0].save())  # [seq,hidden] 또는 [batch,seq,hidden]
        for L in range(nL):
            t = saved[L]
            t = t if isinstance(t, torch.Tensor) else t.value
            last = t[-1] if t.ndim == 2 else t[0, -1]  # 마지막 토큰(모델 간 차원 흡수)
            vec = last.float().detach().cpu().numpy()
            if X is None:
                X = np.zeros((len(items), nL, vec.shape[-1]), dtype=np.float32)
            X[i, L] = vec
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    y = np.array([1 if it["cond"] == "honorific" else 0 for it in items])
    lexeme = np.array([it["lexeme_id"] for it in items])
    axis = np.array([it["axis"] for it in items])
    cond = np.array([it["cond"] for it in items])
    frame = np.array([it["frame_id"] for it in items])
    ocase = np.array([str(it.get("object_case")) for it in items])

    outp = ROOT / f"analysis/output/acts_{key}{suf}.npz"
    np.savez_compressed(outp, X=X, y=y, lexeme=lexeme, axis=axis,
                        cond=cond, frame=frame, object_case=ocase)
    print(f"[{key}] 저장 {outp.relative_to(ROOT)}  X={X.shape}", flush=True)


if __name__ == "__main__":
    main()
