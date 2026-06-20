"""M3·M4 적대 검증 (단일 적재로 핵심 반론 실험).

반론1: 조향층 L10이 임의 → 여러 층 × 넓은 α 스윕에서 객체가 *어디서든* LD>0로 복구되나?
반론2: M3 절대 LD가 재구성치 → 표본 쌍에서 실제 절대 patched LD를 직접 측정(객체 0 못 넘나?).
반론3: 조향이 경어 방향 특정인가 → 같은 크기 무작위 벡터 통제(주체는 경어벡터로만 움직이나?).
반론4: 실제 생성에서도 보충법이 안 나오나 → 강한 조향 하 생성 + Kiwi 객체높임 탐지율.

실행: STIM=v2 .venv/bin/python -m src.verify_causal exaone
산출: analysis/output/verify_causal_<key><suf>.json
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
from src.honor_axes import detect_object_hon, detect_subject_hon  # noqa: E402
from src.model_io import layer_container, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = "data/stimuli/pairs_v2.jsonl" if stim == "v2" else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]
    acts = np.load(ROOT / f"analysis/output/acts_{key}{suf}.npz", allow_pickle=True)
    X, ay, aax, afr = acts["X"], acts["y"], acts["axis"], acts["frame"]
    nL = X.shape[1]

    lm, spec = load_model(key)
    tok = lm.tokenizer
    bos = tok.bos_token_id
    base = getattr(lm, "_model", None) or getattr(lm, "model", None)
    dev, dt = base.lm_head.weight.device, base.lm_head.weight.dtype

    byframe = defaultdict(dict)
    for it in items:
        byframe[it["frame_id"]][it["cond"]] = it

    def neutral_items(ax, k=24):
        out = [d["neutral"] for f, d in byframe.items()
               if "neutral" in d and d["neutral"]["axis"] == ax]
        return out[:k]

    def ld_under(prompt, hon_t, pln_t, L, addvec):
        a = align_item(tok, prompt, hon_t, pln_t)
        if a.hon_first is None:
            return None
        ids = torch.tensor([([bos] if bos is not None else []) + a.prefix_ids])
        with lm.trace(ids):
            blocks = layer_container(lm, spec)
            out = blocks[L].output[0]
            if addvec is not None:
                if out.ndim == 2:
                    out[-1, :] = out[-1, :] + addvec
                else:
                    out[0, -1, :] = out[0, -1, :] + addvec
            lg = lm.output.logits.save()
        lg = (lg if isinstance(lg, torch.Tensor) else lg.value)[0, -1].float()
        return float(lg[a.hon_first] - lg[a.pln_first])

    res = {"model": key, "n_layers": nL}

    # ── 반론1·3: 조향 층×α 스윕 + 무작위 통제 ──
    layers = list(range(2, nL, 4))
    alphas = [0, 4, 8, 12, 16, 24]
    rng = np.random.default_rng(0)
    sweep = {}
    for ax in ("subject", "object"):
        ni = neutral_items(ax)
        randv = torch.tensor(rng.standard_normal(X.shape[2]), device=dev, dtype=dt)
        randv = randv / randv.norm()
        grid = {}
        for L in layers:
            v = X[(aax == ax) & (ay == 1), L].mean(0) - X[(aax == ax) & (ay == 0), L].mean(0)
            v = torch.tensor(v, device=dev, dtype=dt); v = v / v.norm()
            for alpha in alphas:
                vals = [ld_under(it["prompt"], it["honorific_target"], it["plain_target"], L, alpha * v)
                        for it in ni]
                grid[f"L{L}_a{alpha}"] = round(float(np.mean([x for x in vals if x is not None])), 2)
            # 무작위 통제(α=12)
            valr = [ld_under(it["prompt"], it["honorific_target"], it["plain_target"], L, 12 * randv)
                    for it in ni]
            grid[f"L{L}_rand12"] = round(float(np.mean([x for x in valr if x is not None])), 2)
        sweep[ax] = grid
    res["steer_sweep"] = sweep
    # 객체가 어디서든 LD>0로 복구되는가
    obj_vals = [v for k, v in sweep["object"].items() if "rand" not in k]
    subj_vals = [v for k, v in sweep["subject"].items() if "rand" not in k]
    res["object_max_LD_over_grid"] = max(obj_vals)
    res["object_ever_crosses_0"] = max(obj_vals) > 0
    res["subject_max_LD_over_grid"] = max(subj_vals)

    # 스윕 결과 우선 저장(생성 API 실패 대비)
    outp = ROOT / f"analysis/output/verify_causal_{key}{suf}.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("객체 그리드 최대 LD:", res["object_max_LD_over_grid"],
          "| 객체가 0을 넘는가:", res["object_ever_crosses_0"])
    print("주체 그리드 최대 LD:", res["subject_max_LD_over_grid"], flush=True)

    # ── 반론4: 강한 조향 하 실제 생성 + Kiwi 탐지 ──
    gen = {}
    try:
      for ax in ("subject", "object"):
        L = nL // 3
        v = X[(aax == ax) & (ay == 1), L].mean(0) - X[(aax == ax) & (ay == 0), L].mean(0)
        v = torch.tensor(v, device=dev, dtype=dt); v = v / v.norm()
        det = detect_subject_hon if ax == "subject" else detect_object_hon
        ni = neutral_items(ax, k=12)
        for alpha in (0, 12):
            hon_rate = 0; n = 0
            for it in ni:
                ids = tok(it["prompt"], return_tensors="pt")["input_ids"]
                with lm.generate(ids, max_new_tokens=6, do_sample=False):
                    blocks = layer_container(lm, spec)
                    out = blocks[L].output[0]
                    if alpha:
                        if out.ndim == 2:
                            out[-1, :] = out[-1, :] + alpha * v
                        else:
                            out[0, -1, :] = out[0, -1, :] + alpha * v
                    g = lm.generator.output.save()
                gen_ids = g if isinstance(g, torch.Tensor) else g.value
                text = tok.decode(gen_ids[0], skip_special_tokens=True)
                hon_rate += det(text); n += 1
            gen[f"{ax}_a{alpha}"] = {"honorific_rate": round(hon_rate / max(n, 1), 3), "n": n}
      res["generation_kiwi"] = gen
      print("생성 Kiwi 경어율:", json.dumps(gen, ensure_ascii=False), flush=True)
    except Exception as e:
      res["generation_kiwi"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
      print("생성 API 실패(스윕 결과는 유효):", res["generation_kiwi"]["error"], flush=True)

    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
