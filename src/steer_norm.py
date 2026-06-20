"""M4 노름-매칭 조향 (모델 간 공정 비교). 적대 검증 권고: 고정 절대 α는 Qwen(잔차 노름 큼)에
미세 섭동이라 무효 → α를 **해당 층 잔차 노름의 비율**로 정의해 모든 모델·층에 동일 상대강도.

steer 강도 = frac × mean‖resid_L‖ × unit(경어방향). frac ∈ {0.5,1,2}. held-out 중립에 적용.
주체 vs 객체 최대 LD를 모델별로 비교 — 공정 강도에서도 보충법이 취약한가.

실행: STIM=v2 .venv/bin/python -m src.steer_norm <model>
산출: analysis/output/steernorm_<key><suf>.json
"""
from __future__ import annotations

import json
import os
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
FRACS = [0.5, 1.0, 2.0]


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = "data/stimuli/pairs_v2.jsonl" if stim == "v2" else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]
    acts = np.load(ROOT / f"analysis/output/acts_{key}{suf}.npz", allow_pickle=True)
    X, ay, aax, afr = acts["X"], acts["y"], acts["axis"], acts["frame"]
    nL = X.shape[1]
    resid_norm = np.linalg.norm(X, axis=2).mean(0)  # [nL] 층별 평균 잔차 노름

    lm, spec = load_model(key)
    tok = lm.tokenizer
    bos = tok.bos_token_id
    BOS = [bos] if bos is not None else []
    base = getattr(lm, "_model", None) or getattr(lm, "model", None)
    dev, dt = base.lm_head.weight.device, base.lm_head.weight.dtype

    def steer_ld(prompt, hon_t, pln_t, L, vec):
        a = align_item(tok, prompt, hon_t, pln_t)
        if a.hon_first is None:
            return None
        ids = torch.tensor([BOS + a.prefix_ids])
        with lm.trace(ids):
            out = layer_container(lm, spec)[L].output[0]
            if out.ndim == 2:
                out[-1, :] = out[-1, :] + vec
            else:
                out[0, -1, :] = out[0, -1, :] + vec
            lg = lm.output.logits.save()
        lg = (lg if isinstance(lg, torch.Tensor) else lg.value)[0, -1].float()
        return float(lg[a.hon_first] - lg[a.pln_first])

    layers = list(range(2, nL, 5))
    frames = sorted(set(afr.tolist()))
    rng = np.random.default_rng(0)
    folds = [set(np.array(frames)[rng.permutation(len(frames))[i::2]]) for i in range(2)]

    res = {"model": key, "stim": stim or "v1", "fracs": FRACS,
           "resid_norm_by_layer": resid_norm.round(1).tolist()}
    from collections import defaultdict
    sweep = {ax: defaultdict(list) for ax in ("subject", "object")}
    for fold in folds:
        tr = np.array([f not in fold for f in afr])
        for ax in ("subject", "object"):
            ni = [it for i, it in enumerate(items) if it["axis"] == ax
                  and it["cond"] == "neutral" and afr[i] in fold][:16]
            for L in layers:
                vraw = X[tr & (aax == ax) & (ay == 1), L].mean(0) - X[tr & (aax == ax) & (ay == 0), L].mean(0)
                v = torch.tensor(vraw, device=dev, dtype=dt); v = v / v.norm()
                for frac in FRACS:
                    mag = frac * float(resid_norm[L])
                    for it in ni:
                        ld = steer_ld(it["prompt"], it["honorific_target"], it["plain_target"], L, mag * v)
                        if ld is not None:
                            sweep[ax][f"L{L}_f{frac}"].append(ld)
    for ax in ("subject", "object"):
        g = {k: round(float(np.mean(v)), 2) for k, v in sweep[ax].items()}
        res[ax] = {"grid": g, "max_LD": round(max(g.values()), 2),
                   "cells_cross0": sum(v > 0 for v in g.values()), "n_cells": len(g)}
    outp = ROOT / f"analysis/output/steernorm_{key}{suf}.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{key}] 노름-매칭 조향 최대 LD: 주체={res['subject']['max_LD']} "
          f"객체={res['object']['max_LD']} (객체 0넘는셀 {res['object']['cells_cross0']}/{res['object']['n_cells']})",
          flush=True)
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
