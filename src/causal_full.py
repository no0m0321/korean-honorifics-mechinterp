"""엄밀 인과 측정 (M3 직접 + M4 통제 + 어휘별) — 적대 검증 권고 반영.

M3-직접: 결정 지점 잔차를 clean→corrupt 치환하되 per-pair **절대** patched LD를 직접 측정
  (기존 patch.py는 정규화 회복률만 저장 → 절대 LD는 재구성이었음). 축별 mean과 '0 넘는 쌍 비율'.
M4-통제: 조향 층×α 스윕 + 통제 (a) cross-axis(다른 축 벡터로 밀기), (b) shuffled-label 벡터.
  무작위 대신 의미 있는 통제로 '객체 벡터 결함 vs 어휘 도달불가' 분리. 객체 6어휘별 분해.

단일 적재. 순환 방지: 조향 벡터는 train 프레임, 적용은 held-out 중립.
실행: STIM=v2 .venv/bin/python -m src.causal_full <model>
산출: analysis/output/causal_<key><suf>.json
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
    BOS = [bos] if bos is not None else []

    byframe = defaultdict(dict)
    for it in items:
        byframe[it["frame_id"]][it["cond"]] = it
    pairs = [(f, d["honorific"], d["neutral"]) for f, d in byframe.items()
             if "honorific" in d and "neutral" in d]
    res = {"model": key, "stim": stim or "v1", "n_layers": nL}

    # ───────── M3 직접: per-pair 절대 patched LD ─────────
    def ld_at(logits, hf, pf):
        lg = (logits if isinstance(logits, torch.Tensor) else logits.value)[0, -1].float()
        return float(lg[hf] - lg[pf])

    abs_patched = {"subject": [], "object": []}  # 각 원소: [nL] 절대 patched LD
    cleancorr = {"subject": [], "object": []}
    for frame, clean_it, corr_it in pairs:
        a_c = align_item(tok, clean_it["prompt"], clean_it["honorific_target"], clean_it["plain_target"])
        a_k = align_item(tok, corr_it["prompt"], corr_it["honorific_target"], corr_it["plain_target"])
        if a_c.hon_first is None or a_c.hon_first != a_k.hon_first or a_c.pln_first != a_k.pln_first:
            continue
        hf, pf = a_c.hon_first, a_c.pln_first
        clean_ids = torch.tensor([BOS + a_c.prefix_ids])
        corr_ids = torch.tensor([BOS + a_k.prefix_ids])
        cres = []
        with lm.trace(clean_ids):
            blk = layer_container(lm, spec)
            for L in range(nL):
                cres.append(blk[L].output[0].save())
            clog = lm.output.logits.save()
        cvec = []
        for L in range(nL):
            t = cres[L]; t = t if isinstance(t, torch.Tensor) else t.value
            cvec.append((t[-1] if t.ndim == 2 else t[0, -1]).detach())
        clean_LD = ld_at(clog, hf, pf)
        with lm.trace(corr_ids):
            klog = lm.output.logits.save()
        corr_LD = ld_at(klog, hf, pf)
        row = np.zeros(nL)
        for L in range(nL):
            with lm.trace(corr_ids):
                blk = layer_container(lm, spec)
                out = blk[L].output[0]
                if out.ndim == 2:
                    out[-1, :] = cvec[L]
                else:
                    out[0, -1, :] = cvec[L]
                pl = lm.output.logits.save()
            row[L] = ld_at(pl, hf, pf)
        ax = clean_it["axis"]
        abs_patched[ax].append(row)
        cleancorr[ax].append((clean_LD, corr_LD))
    m3 = {}
    for ax in ("subject", "object"):
        arr = np.array(abs_patched[ax])  # [n, nL] 절대 patched LD
        m3[ax] = {
            "n": len(arr),
            "mean_abs_patched_LD_by_layer": np.nanmean(arr, axis=0).round(2).tolist(),
            "frac_pairs_cross0_by_layer": (arr > 0).mean(axis=0).round(3).tolist(),
            "max_mean_abs_patched_LD": round(float(np.nanmean(arr, axis=0).max()), 2),
            "any_layer_mean_cross0": bool(np.nanmean(arr, axis=0).max() > 0),
            "clean_LD_mean": round(float(np.mean([c for c, _ in cleancorr[ax]])), 2),
            "corr_LD_mean": round(float(np.mean([k for _, k in cleancorr[ax]])), 2),
        }
    res["M3_direct"] = m3
    res_path = ROOT / f"analysis/output/causal_{key}{suf}.json"
    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M3직접] subject 최대 mean patched LD={m3['subject']['max_mean_abs_patched_LD']} "
          f"object={m3['object']['max_mean_abs_patched_LD']} "
          f"(object 어느층이든 mean>0: {m3['object']['any_layer_mean_cross0']})", flush=True)

    # ───────── M4: 조향 층×α + 통제 + 어휘별 ─────────
    def neutral_test(ax, fold, k=16):
        out = [(i, it) for i, it in enumerate(items)
               if it["axis"] == ax and it["cond"] == "neutral" and afr[i] in fold]
        return out[:k]

    def steer_ld(prompt, hon_t, pln_t, L, vec):
        a = align_item(tok, prompt, hon_t, pln_t)
        if a.hon_first is None:
            return None
        ids = torch.tensor([BOS + a.prefix_ids])
        with lm.trace(ids):
            blk = layer_container(lm, spec)
            out = blk[L].output[0]
            if out.ndim == 2:
                out[-1, :] = out[-1, :] + vec
            else:
                out[0, -1, :] = out[0, -1, :] + vec
            lg = lm.output.logits.save()
        return ld_at(lg, a.hon_first, a.pln_first)

    layers = list(range(2, nL, 5))
    alphas = [8, 12, 16]
    Lmid = min(layers, key=lambda L: abs(L - nL // 3))  # 통제는 중간층 한 곳에서만
    frames = sorted(set(afr.tolist()))
    rng = np.random.default_rng(0)
    folds = [set(np.array(frames)[rng.permutation(len(frames))[i::2]]) for i in range(2)]

    def unit(vnp):
        v = torch.tensor(vnp, device=dev, dtype=dt)
        return v / v.norm()

    m4 = {"layers": layers, "alphas": alphas}
    # 메인 + 통제: honorific 벡터, cross-axis 벡터, shuffled-label 벡터
    sweeps = {ax: {"honor": defaultdict(list), "cross": defaultdict(list),
                   "shuf": defaultdict(list)} for ax in ("subject", "object")}
    perlex = defaultdict(lambda: defaultdict(list))  # object 어휘별 [(L,a)] LD
    for fold in folds:
        tr = np.array([f not in fold for f in afr])
        vecs = {}
        for ax in ("subject", "object"):
            for L in layers:
                vecs[(ax, L, "honor")] = unit(X[tr & (aax == ax) & (ay == 1), L].mean(0)
                                              - X[tr & (aax == ax) & (ay == 0), L].mean(0))
                yl = ay[tr & (aax == ax)].copy()
                rng.shuffle(yl)
                Xa = X[tr & (aax == ax), L]
                vecs[(ax, L, "shuf")] = unit(Xa[yl == 1].mean(0) - Xa[yl == 0].mean(0))
        for ax in ("subject", "object"):
            other = "object" if ax == "subject" else "subject"
            for L in layers:
                vecs[(ax, L, "cross")] = vecs[(other, L, "honor")]
            for i, it in neutral_test(ax, fold):
                for L in layers:
                    for a in alphas:
                        controls = ("honor", "cross", "shuf") if L == Lmid else ("honor",)
                        for ctrl in controls:
                            ld = steer_ld(it["prompt"], it["honorific_target"],
                                          it["plain_target"], L, a * vecs[(ax, L, ctrl)])
                            if ld is not None:
                                sweeps[ax][ctrl][f"L{L}_a{a}"].append(ld)
                                if ax == "object" and ctrl == "honor":
                                    perlex[it["lexeme_id"]][f"L{L}_a{a}"].append(ld)
    for ax in ("subject", "object"):
        m4[ax] = {ctrl: {k: round(float(np.mean(v)), 2) for k, v in sweeps[ax][ctrl].items()}
                  for ctrl in ("honor", "cross", "shuf")}
        hon_vals = list(m4[ax]["honor"].values())
        m4[ax]["honor_max_LD"] = round(max(hon_vals), 2)
        m4[ax]["honor_cells_cross0"] = sum(v > 0 for v in hon_vals)
        m4[ax]["n_cells"] = len(hon_vals)
    # object 어휘별 최대 LD
    m4["object_per_lexeme_max_LD"] = {
        lx: round(max(float(np.mean(v)) for v in cells.values()), 2)
        for lx, cells in perlex.items()}
    res["M4_controlled"] = m4
    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M4] subject honor 최대 LD={m4['subject']['honor_max_LD']} "
          f"object honor 최대 LD={m4['object']['honor_max_LD']} "
          f"(0넘는 셀 {m4['object']['honor_cells_cross0']}/{m4['object']['n_cells']})", flush=True)
    print(f"[M4 통제] subject cross 최대="
          f"{max(m4['subject']['cross'].values()):.2f} shuf 최대={max(m4['subject']['shuf'].values()):.2f}",
          flush=True)
    print(f"[M4 어휘별 object 최대 LD] {json.dumps(m4['object_per_lexeme_max_LD'], ensure_ascii=False)}",
          flush=True)
    print("저장:", res_path.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
