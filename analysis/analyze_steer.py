"""4단계 CAA 조향 곡선 (M4): 조향 계수 α에 따른 logit-diff. 주체 vs 객체.

M4 수정 비대칭: 전역 경어 벡터로 -시-(굴절)는 LD가 0을 넘어 복구되나, 보충법은 0을 못 넘음.

실행: STIM=v2 .venv/bin/python analysis/analyze_steer.py <model>
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
for c in ["AppleGothic", "AppleSDGothicNeo", "NanumGothic"]:
    if any(c in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = c
        break
plt.rcParams["axes.unicode_minus"] = False

key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
suf = f"_{os.environ.get('STIM', '')}" if os.environ.get("STIM") else ""
d = json.loads((ROOT / f"analysis/output/steer_{key}{suf}.json").read_text())
alphas = d["alphas"]

fig, ax = plt.subplots(figsize=(7.5, 5))
for axis, col, lab in [("subject", "#2b6cb0", "주체 -시- (굴절)"),
                       ("object", "#c53030", "객체 보충법")]:
    ys = [d[axis][str(a)] for a in alphas]
    ax.plot(alphas, ys, "-o", color=col, label=f"{lab} (n={d[axis]['n']})")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.fill_between(alphas, 0, max(d["subject"][str(alphas[-1])], 0.1), alpha=0.05, color="#2b6cb0")
ax.set_xlabel("조향 계수 α (경어 방향 강도)")
ax.set_ylabel("logit-diff (경어형 − 평형형)")
ax.set_title(f"{key.upper()}: CAA 조향 수정 비대칭 (M4)\n"
             f"전역 경어 벡터(L{d['steer_layer']['subject']})로 -시-는 복구(>0), 보충법은 실패(<0)",
             fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / f"analysis/output/steer_{key}{suf}.png", dpi=150)
print("저장:", f"analysis/output/steer_{key}{suf}.png")
