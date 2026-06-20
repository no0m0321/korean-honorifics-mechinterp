"""v2 모델 계열 비교(M6): M1 명사분리 일반화 + M2 굴절-보충 해리.

rigor_<key>_v2.json 이 있는 모델만 사용(점진적). 두 패널:
  좌) M1 명사분리 정확도(주체/객체) — 경어 요구가 새 인물로 일반화되는 추상 표상인가
  우) M2 최종 logit_diff(주체/객체) — 굴절은 추월(>0), 보충은 추월 실패(<0)

실행: .venv/bin/python analysis/compare_v2.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/output"
for c in ["AppleGothic", "AppleSDGothicNeo", "NanumGothic"]:
    if any(c in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = c
        break
plt.rcParams["axes.unicode_minus"] = False

KEYS = ["exaone", "llama", "qwen"]
NAMES = {"exaone": "EXAONE\n(한국어)", "llama": "Llama\n(영어)", "qwen": "Qwen\n(다국어)"}

avail = [k for k in KEYS if (OUT / f"rigor_{k}_v2.json").exists()]
data = {k: json.loads((OUT / f"rigor_{k}_v2.json").read_text()) for k in avail}
print("모델:", avail)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
x = np.arange(len(avail))
w = 0.36

# 좌: M1 명사분리 일반화
subj_nd = [data[k]["M1"]["subject"]["noun_disjoint_task_acc"] for k in avail]
obj_nd = [data[k]["M1"]["object_pooled"]["noun_disjoint_task_acc"] for k in avail]
ax1.bar(x - w / 2, subj_nd, w, label="주체 -시-", color="#2b6cb0")
ax1.bar(x + w / 2, obj_nd, w, label="객체 보충법", color="#c53030")
ax1.axhline(0.5, ls="--", color="gray", lw=1, label="우연")
ax1.set_xticks(x); ax1.set_xticklabels([NAMES[k] for k in avail])
ax1.set_ylim(0, 1.05); ax1.set_ylabel("명사분리 정확도(새 인물 일반화)")
ax1.set_title("M1: 경어 요구의 추상 인코딩", fontsize=11)
ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(subj_nd, obj_nd)):
    ax1.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    ax1.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

# 우: M2 최종 logit_diff
subj_ld = [data[k]["M2"]["subject_final_LD_mean"] for k in avail]
obj_ld = [data[k]["M2"]["object_final_LD_mean"] for k in avail]
ax2.bar(x - w / 2, subj_ld, w, label="주체 -시-", color="#2b6cb0")
ax2.bar(x + w / 2, obj_ld, w, label="객체 보충법", color="#c53030")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels([NAMES[k] for k in avail])
ax2.set_ylabel("최종 logit_diff (경어형 − 평형형)")
ax2.set_title("M2: 굴절은 추월(>0), 보충은 실패(<0)", fontsize=11)
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(subj_ld, obj_ld)):
    ax2.text(i - w / 2, a + (0.1 if a >= 0 else -0.3), f"+{a:.1f}", ha="center", fontsize=8)
    ax2.text(i + w / 2, b - 0.3, f"{b:.1f}", ha="center", fontsize=8)

fig.suptitle("교란 통제 v2: 경어 요구는 추상 인코딩(좌), 보충법만 추월 실패(우)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "compare_v2.png", dpi=150)
print("저장: analysis/output/compare_v2.png")

# 표
print(f"\n{'모델':10}{'주체 명사분리':>12}{'객체 명사분리':>12}{'주체 최종LD':>12}{'객체 최종LD':>12}{'Cohen d':>9}")
for k in avail:
    m1, m2 = data[k]["M1"], data[k]["M2"]
    print(f"{k:10}{m1['subject']['noun_disjoint_task_acc']:>12}"
          f"{m1['object_pooled']['noun_disjoint_task_acc']:>12}"
          f"{m2['subject_final_LD_mean']:>12}{m2['object_final_LD_mean']:>12}{m2['cohen_d_lexeme']:>9}")
