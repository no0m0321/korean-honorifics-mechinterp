"""4단계 CAA 조향 (가설 M4): 인과 수정.

경어−중립 대조쌍의 잔차 평균차로 '경어 방향' 조향 벡터를 만들어(Rimsky et al. 2024),
중립 프롬프트에 추론 시 더하고 출력이 경어형으로 가는지(logit-diff 변화) 측정한다.
순환 방지: 조향 벡터는 train 프레임에서, 적용은 held-out 중립 항목에서(프레임 단위 분할).

예측(M4): -시-(굴절) 누락은 전역 경어 벡터로 복구되나, 보충법 어휘는 복구 안 됨(전역 방향이
항목 특정 어휘를 소환 못함) → 수정 비대칭. logit-diff가 주체는 0 위로, 객체는 못 넘음.

캐시된 잔차(acts_<key><suf>.npz, 동사직전 위치)로 조향 벡터 산출, 모델로 중립에 적용.
검증: STIM=v2 .venv/bin/python -m src.steer <model> test
실행: STIM=v2 .venv/bin/python -m src.steer <model>
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.align import align_item  # noqa: E402
from src.model_io import layer_container, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = [0.0, 4.0, 8.0, 12.0]


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    test = len(sys.argv) > 2 and sys.argv[2] == "test"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = "data/stimuli/pairs_v2.jsonl" if stim == "v2" else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]
    acts = np.load(ROOT / f"analysis/output/acts_{key}{suf}.npz", allow_pickle=True)
    X, ay, aax, afr = acts["X"], acts["y"], acts["axis"], acts["frame"]
    nL = X.shape[1]

    # 항목 인덱스(npz 행 = items 순서)
    lm, spec = load_model(key)
    tok = lm.tokenizer
    bos = tok.bos_token_id
    base = getattr(lm, "_model", None) or getattr(lm, "model", None)
    head_w = base.lm_head.weight
    dev, dt = head_w.device, head_w.dtype
    # 조향 층: 각 축 중기(국소 신호 가설). 단일 층 + α 스윕.
    Lsteer = {"subject": nL // 3, "object": nL // 3}
    print(f"[{key}] 조향층 {Lsteer} layers={nL}", flush=True)

    # 프레임 단위 5분할(순환 방지): train으로 조향벡터, test 중립에 적용
    frames = sorted(set(afr.tolist()))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(frames))
    folds = [set(np.array(frames)[perm[i::5]]) for i in range(5)]

    results = {ax: {a: [] for a in ALPHAS} for ax in ("subject", "object")}
    for fold in (folds[:1] if test else folds):
        train_mask = np.array([f not in fold for f in afr])
        for ax in ("subject", "object"):
            L = Lsteer[ax]
            tm = train_mask & (aax == ax)
            vpos = X[tm & (ay == 1), L].mean(0)
            vneg = X[tm & (ay == 0), L].mean(0)
            v = torch.tensor(vpos - vneg, device=dev, dtype=dt)
            v = v / v.norm()
            # held-out 중립 항목
            test_items = [it for i, it in enumerate(items)
                          if it["axis"] == ax and it["cond"] == "neutral"
                          and afr[i] in fold]
            if test:
                test_items = test_items[:3]
            for it in test_items:
                a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
                if a.hon_first is None:
                    continue
                ids = torch.tensor([([bos] if bos is not None else []) + a.prefix_ids])
                for alpha in ALPHAS:
                    with lm.trace(ids):
                        blocks = layer_container(lm, spec)
                        out = blocks[L].output[0]
                        add = alpha * v
                        if out.ndim == 2:
                            out[-1, :] = out[-1, :] + add
                        else:
                            out[0, -1, :] = out[0, -1, :] + add
                        lg = lm.output.logits.save()
                    lg = (lg if isinstance(lg, torch.Tensor) else lg.value)[0, -1].float()
                    results[ax][alpha].append(float(lg[a.hon_first] - lg[a.pln_first]))
        if test:
            break

    out = {"model": key, "stim": stim or "v1", "steer_layer": Lsteer, "alphas": ALPHAS}
    for ax in ("subject", "object"):
        out[ax] = {str(a): round(float(np.mean(results[ax][a])), 3) for a in ALPHAS}
        out[ax]["n"] = len(results[ax][ALPHAS[0]])
    if test:
        print(json.dumps(out, ensure_ascii=False), flush=True)
        print("test OK", flush=True)
        return
    outp = ROOT / f"analysis/output/steer_{key}{suf}.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for ax in ("subject", "object"):
        prog = " → ".join(f"α{a}:{out[ax][str(a)]:+.2f}" for a in ALPHAS)
        print(f"[{key}/{ax}] logit-diff  {prog}  (n={out[ax]['n']})", flush=True)
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
