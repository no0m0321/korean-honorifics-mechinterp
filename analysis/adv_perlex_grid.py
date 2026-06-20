"""적대 재검토: M4 객체 조향의 어휘별 전체 그리드를 같은 (L,a) 셀 기준으로 재집계.
causal_full.py의 M4 객체 honor 경로를 정확히 복제하되, 어휘별 셀 평균을 모두 저장.
산출: 같은 셀에서 5어휘 동시 양수 셀이 존재하는가? 어휘별 argmax 셀 분산은?
실행: STIM=v2 .venv/bin/python analysis/adv_perlex_grid.py <model>
"""
from __future__ import annotations
import json, os, sys, warnings
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch
from src.align import align_item
from src.model_io import layer_container, load_model

ROOT = Path(__file__).resolve().parent.parent

def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    stim = os.environ.get("STIM", "v2")
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

    def ld_at(logits, hf, pf):
        lg = (logits if isinstance(logits, torch.Tensor) else logits.value)[0, -1].float()
        return float(lg[hf] - lg[pf])

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
    frames = sorted(set(afr.tolist()))
    rng = np.random.default_rng(0)
    folds = [set(np.array(frames)[rng.permutation(len(frames))[i::2]]) for i in range(2)]

    def unit(vnp):
        v = torch.tensor(vnp, device=dev, dtype=dt)
        return v / v.norm()

    def neutral_test(ax, fold, k=16):
        out = [(i, it) for i, it in enumerate(items)
               if it["axis"] == ax and it["cond"] == "neutral" and afr[i] in fold]
        return out[:k]

    ax = "object"
    # 어휘별·셀별 LD 누적
    perlex = defaultdict(lambda: defaultdict(list))   # lexeme -> cell -> [LD]
    agg = defaultdict(list)                            # cell -> [LD]  (전체 집계, 검증용)
    for fold in folds:
        tr = np.array([f not in fold for f in afr])
        vecs = {}
        for L in layers:
            vecs[L] = unit(X[tr & (aax == ax) & (ay == 1), L].mean(0)
                           - X[tr & (aax == ax) & (ay == 0), L].mean(0))
        for i, it in neutral_test(ax, fold):
            for L in layers:
                for a in alphas:
                    ld = steer_ld(it["prompt"], it["honorific_target"],
                                  it["plain_target"], L, a * vecs[L])
                    if ld is not None:
                        cell = f"L{L}_a{a}"
                        perlex[it["lexeme_id"]][cell].append(ld)
                        agg[cell].append(ld)

    lexemes = sorted(perlex.keys())
    cells = [f"L{L}_a{a}" for L in layers for a in alphas]
    # 어휘×셀 평균 행렬
    grid = {lx: {c: round(float(np.mean(perlex[lx][c])), 3) for c in cells} for lx in lexemes}
    agg_grid = {c: round(float(np.mean(agg[c])), 3) for c in cells}

    # Q1 핵심: 모든 어휘가 동시에 양수인 셀
    joint_pos_cells = [c for c in cells if all(grid[lx][c] > 0 for lx in lexemes)]
    # 셀별 양수 어휘 수
    cell_npos = {c: sum(grid[lx][c] > 0 for lx in lexemes) for c in cells}
    # 어휘별 argmax 셀 (max-over-cells가 어디서 나오나)
    lex_argmax = {lx: max(cells, key=lambda c: grid[lx][c]) for lx in lexemes}
    lex_max = {lx: round(grid[lx][lex_argmax[lx]], 3) for lx in lexemes}
    # 각 어휘의 '최적 공통셀'(집계 argmax)에서의 값
    best_agg_cell = max(cells, key=lambda c: agg_grid[c])
    lex_at_best_agg = {lx: grid[lx][best_agg_cell] for lx in lexemes}

    out = {
        "model": key, "stim": stim, "axis": ax,
        "layers": layers, "alphas": alphas, "lexemes": lexemes,
        "n_per_cell_example": {c: len(agg[c]) for c in cells[:1]},
        "grid_per_lexeme": grid,
        "agg_grid": agg_grid,
        "best_agg_cell": best_agg_cell,
        "best_agg_value": agg_grid[best_agg_cell],
        "lex_value_at_best_agg_cell": {lx: round(v, 3) for lx, v in lex_at_best_agg.items()},
        "n_lex_positive_at_best_agg_cell": sum(v > 0 for v in lex_at_best_agg.values()),
        "lex_argmax_cell": lex_argmax,
        "lex_max_over_cells": lex_max,
        "n_distinct_argmax_cells": len(set(lex_argmax.values())),
        "joint_all_positive_cells": joint_pos_cells,
        "max_npos_per_cell": max(cell_npos.values()),
        "cell_npos": cell_npos,
    }
    outp = ROOT / f"analysis/output/adv_perlex_{key}{suf}.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{key}] best agg cell={best_agg_cell}({agg_grid[best_agg_cell]}) "
          f"lex positive there={out['n_lex_positive_at_best_agg_cell']}/{len(lexemes)}")
    print(f"  joint all-positive cells: {joint_pos_cells if joint_pos_cells else 'NONE'}")
    print(f"  max #lex positive in any single cell: {out['max_npos_per_cell']}/{len(lexemes)}")
    print(f"  distinct argmax cells across lexemes: {out['n_distinct_argmax_cells']}/{len(lexemes)}")
    print(f"  lex argmax cells: {lex_argmax}")
    print("saved:", outp.relative_to(ROOT))

if __name__ == "__main__":
    main()
