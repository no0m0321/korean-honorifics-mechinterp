"""2단계 로짓 렌즈 분석·시각화 (M2: 추월층).

축(subject/object) × 조건(honorific/neutral)별 층별 평균 logit_diff 곡선을 그리고,
추월층(평균 logit_diff가 음→양으로 전환하는 첫 층)을 추정한다. honorific 조건에서
주체는 이른 추월층, 객체는 늦거나 약한 추월을 예측(M2).

실행: .venv/bin/python analysis/analyze_lens.py exaone
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
for cand in ["AppleGothic", "AppleSDGothicNeo", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False


def crossover(mean_curve):
    """평균 곡선이 음→양으로 처음 바뀌고 이후 유지되는 층. 없으면 None."""
    for L in range(len(mean_curve)):
        if mean_curve[L] > 0 and all(mean_curve[k] > 0 for k in range(L, len(mean_curve))):
            return L
    return None


def main():
    import os
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    suf = f"_{os.environ.get('STIM', '')}" if os.environ.get("STIM") else ""
    z = np.load(ROOT / f"analysis/output/lens_{key}{suf}.npz", allow_pickle=True)
    diff, axis, cond = z["diff"], z["axis"], z["cond"]
    nL = diff.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {("subject", "honorific"): ("#2b6cb0", "-", "주체 -시- (경어촉발)"),
              ("subject", "neutral"): ("#2b6cb0", ":", "주체 -시- (중립)"),
              ("object", "honorific"): ("#c53030", "-", "객체 보충법 (경어촉발)"),
              ("object", "neutral"): ("#c53030", ":", "객체 보충법 (중립)")}
    summary = {"model": key, "n_layers": int(nL)}
    for (ax_name, c), (col, ls, lab) in styles.items():
        mask = (axis == ax_name) & (cond == c)
        curve = np.nanmean(diff[mask], axis=0)
        ax.plot(range(nL), curve, ls=ls, color=col, lw=2, label=lab)
        co = crossover(curve)
        summary[f"{ax_name}_{c}"] = {
            "final_logit_diff": round(float(curve[-1]), 3),
            "crossover_layer": co,
            "max_logit_diff": round(float(np.max(curve)), 3),
        }
        if c == "honorific" and co is not None:
            ax.axvline(co, color=col, ls="--", alpha=0.4)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("층 (layer)")
    ax.set_ylabel("logit_diff = logit(경어형) - logit(평형형)")
    ax.set_title(f"{key.upper()}: 로짓 렌즈 추월층 (M2)\n"
                 "주체 -시-는 후기층 확실 추월, 객체 보충법은 약한 추월", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outp = ROOT / f"analysis/output/lens_{key}{suf}.png"
    fig.savefig(outp, dpi=150)

    (ROOT / f"analysis/output/lens_{key}{suf}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
