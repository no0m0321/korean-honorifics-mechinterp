"""3단계 패칭 회복 곡선 (M3). 층별 logit-diff 회복률: 주체 vs 객체.

M3 예측: -시-(굴절)는 더 이른/중기 층 치환으로 회복(국소 신호), 보충법은 더 늦게/덜 회복.
회복이 0.5를 처음 넘는 층 = 그 신호가 결정 지점에 확립되는 깊이.

실행: STIM=v2 .venv/bin/python analysis/analyze_patch.py <model>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
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
d = json.loads((ROOT / f"analysis/output/patch_{key}{suf}.json").read_text())
nL = d["n_layers"]
# clean(honorific)/corrupt(neutral) 절대 logit-diff 평균(렌즈 요약과 동일 측정점)
lens = json.loads((ROOT / f"analysis/output/lens_{key}{suf}_summary.json").read_text())
absld = {"subject": (lens["subject_neutral"]["final_logit_diff"],
                     lens["subject_honorific"]["final_logit_diff"]),
         "object": (lens["object_neutral"]["final_logit_diff"],
                    lens["object_honorific"]["final_logit_diff"])}


def first_cross(curve, thr):
    for L, v in enumerate(curve):
        if v >= thr:
            return L
    return None


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
summary = {}
for axis, col, lab in [("subject", "#2b6cb0", "주체 -시- (굴절)"),
                       ("object", "#c53030", "객체 보충법")]:
    rec = np.array(d[axis]["mean_recovery_by_layer"])
    corr, clean = absld[axis]
    abs_ld = corr + rec * (clean - corr)  # 절대 logit-diff(치환 층별, 집계 재구성)
    ax1.plot(range(nL), rec, "-o", ms=3, color=col, label=f"{lab} (n={d[axis]['n']})")
    ax2.plot(range(nL), abs_ld, "-o", ms=3, color=col, label=lab)
    cross0 = first_cross(abs_ld, 0.0)  # 경어형이 평형형을 추월(LD>0)하는 치환 층
    summary[axis] = {"recovery_first_cross_0.5": first_cross(rec, 0.5),
                     "clean_LD": clean, "corrupt_LD": corr,
                     "abs_LD_crosses_0_at_layer": cross0,
                     "full_patch_abs_LD": round(float(abs_ld[-1]), 2)}

ax1.axhline(0.5, color="gray", ls=":", lw=1); ax1.axhline(1.0, color="gray", ls=":", lw=0.5)
ax1.set_xlabel("치환 층"); ax1.set_ylabel("회복률 (clean으로 복원)")
ax1.set_title("정규화 회복률 — 둘 다 clean으로 복원", fontsize=10)
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.axhline(0, color="black", lw=0.8)
ax2.set_xlabel("치환 층"); ax2.set_ylabel("절대 logit-diff (경어형 − 평형형)")
ax2.set_title("절대 LD — 주체만 경어형 유도(>0), 객체는 끝내 평형(<0)", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle(f"{key.upper()}: 활성값 패칭 (M3) — 결정 지점 잔차를 clean으로 치환", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / f"analysis/output/patch_{key}{suf}.png", dpi=150)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("저장:", f"analysis/output/patch_{key}{suf}.png")
