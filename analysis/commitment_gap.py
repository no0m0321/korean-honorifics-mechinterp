"""첫 토큰 commitment 병목의 직접 증거 (핵심 기전).

가설: 보충법 실패의 상당 부분은 '첫 분기 토큰 commitment'이다. 모델은 보충법 전체 형태를
적절히 평가하나(시퀀스 로그우도>0), 첫 분기 토큰에서 평형형에 commit한다(첫토큰 logit-diff<0).
greedy 디코딩은 첫 토큰을 따르므로 평형형을 생성한다.

항목을 4사분면으로 분류(honorific 조건):
  commit_fail : 첫토큰 LD<0 AND 전체형 LL>0  ← 병목 서명(전체형은 알지만 첫토큰서 commit 실패)
  both_fail   : 첫토큰<0 AND 전체형<0          ← 진성 실패(전체형도 평형 선호)
  both_succeed: 첫토큰>0 AND 전체형>0
  ft_only     : 첫토큰>0 AND 전체형<0

입력: lens_<k>_v2.npz(첫토큰 LD by layer), seqm2_<k>_v2.npz(전체형 token-mean LL diff).
실행: .venv/bin/python analysis/commitment_gap.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/output"
KEYS = ["exaone", "llama", "qwen"]

print("첫 토큰 commitment 병목 — honorific 항목 4사분면 분류\n")
summary = {}
for k in KEYS:
    lens = np.load(OUT / f"lens_{k}_v2.npz", allow_pickle=True)
    seq = np.load(OUT / f"seqm2_{k}_v2.npz", allow_pickle=True)
    ft_all = lens["diff"][:, -1]          # 최종층 첫토큰 LD
    sq_all = seq["diff"]                  # 전체형 token-mean LL diff
    axis, cond = lens["axis"], lens["cond"]
    # 정렬 확인: 두 npz는 같은 pairs_v2 순서
    assert len(ft_all) == len(sq_all), f"{k} 길이 불일치"
    summary[k] = {}
    for ax in ("subject", "object"):
        m = (axis == ax) & (cond == "honorific") & np.isfinite(ft_all) & np.isfinite(sq_all)
        ft, sq = ft_all[m], sq_all[m]
        n = len(ft)
        cells = {
            "commit_fail": int(((ft < 0) & (sq > 0)).sum()),
            "both_fail": int(((ft < 0) & (sq < 0)).sum()),
            "both_succeed": int(((ft > 0) & (sq > 0)).sum()),
            "ft_only": int(((ft > 0) & (sq < 0)).sum()),
        }
        frac = {c: round(v / n, 3) for c, v in cells.items()}
        # 첫토큰 실패 항목 중 전체형은 성공인 비율 = commitment 병목의 몫
        ft_fail = cells["commit_fail"] + cells["both_fail"]
        commit_share = round(cells["commit_fail"] / ft_fail, 3) if ft_fail else None
        summary[k][ax] = {"n": n, "cells": cells, "frac": frac,
                          "commit_share_of_ft_failures": commit_share,
                          "ft_mean": round(float(ft.mean()), 2), "sq_mean": round(float(sq.mean()), 2)}
        print(f"[{k}/{ax}] n={n}  첫토큰평균={ft.mean():+.2f} 전체형평균={sq.mean():+.2f}")
        print(f"   commit_fail(첫토큰만 실패) {frac['commit_fail']}  both_fail {frac['both_fail']}  "
              f"both_succeed {frac['both_succeed']}  ft_only {frac['ft_only']}")
        if commit_share is not None:
            print(f"   → 첫토큰 실패 항목의 {commit_share:.0%}가 '전체형은 경어형 선호'(commitment 병목)")
    print()

(OUT / "commitment_gap.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장: analysis/output/commitment_gap.json")
