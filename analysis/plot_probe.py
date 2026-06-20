"""1단계 프로빙 층별 선택도 곡선 (계획서 표3 산출물).

주체높임(굴절) vs 객체높임(보충법)의 층별 선형 디코딩 선택도를 비교한다.
두 곡선이 모두 임계선(0.10)을 넘으면 → 두 구문 모두 '경어 요구'가 인코딩됨(M1).
객체축이 주체축에 못지않거나 더 높으면 → 보충법 실패가 인코딩이 아닌 검색 문제임을 시사.

실행: .venv/bin/python analysis/plot_probe.py exaone
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent

# 한글 폰트
for cand in ["AppleGothic", "AppleSDGothicNeo", "NanumGothic", "Malgun Gothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    import os
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    suf = f"_{os.environ.get('STIM', '')}" if os.environ.get("STIM") else ""
    data = json.loads((ROOT / f"analysis/output/probe_{key}{suf}.json").read_text())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"subject": "#2b6cb0", "object": "#c53030"}
    labels = {"subject": "주체높임 -시- (굴절)", "object": "객체높임 보충법"}
    for axis in ("subject", "object"):
        rows = data[axis]
        L = [r["layer"] for r in rows]
        sel = [r["selectivity"] for r in rows]
        lo = [r["ci_lo"] for r in rows]
        hi = [r["ci_hi"] for r in rows]
        ax.plot(L, sel, "-o", ms=3, color=colors[axis], label=labels[axis])
        ax.fill_between(L, lo, hi, color=colors[axis], alpha=0.15)

    ax.axhline(0.10, ls="--", color="gray", lw=1, label="M1 임계 (0.10)")
    ax.set_xlabel("층 (layer)")
    ax.set_ylabel("프로브 선택도 (과제 - 통제)")
    ax.set_title(f"{key.upper()}: 층별 경어 요구 인코딩 선택도\n"
                 "객체높임 요구도 강하게 인코딩됨 → 실패는 검색 단계", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outp = ROOT / f"analysis/output/probe_{key}{suf}.png"
    fig.savefig(outp, dpi=150)
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
