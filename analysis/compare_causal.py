"""3모델 인과 비교 (M3 직접 + M4 통제). 인과 해리의 모델 보편성(M6).

causal_<key>_v2.json 이 있는 모델만. 핵심: M4에서 주체 honor 최대 LD vs 객체 honor 최대 LD,
그리고 통제(cross/shuf)와 객체 어휘별 분해. 그림: 모델별 주체/객체 최대 조향 LD.

실행: .venv/bin/python analysis/compare_causal.py
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
avail = [k for k in KEYS if (OUT / f"causal_{k}_v2.json").exists()]
data = {k: json.loads((OUT / f"causal_{k}_v2.json").read_text()) for k in avail}
print("인과 측정 가능 모델:", avail)

rows = []
for k in avail:
    d = data[k]
    m3, m4 = d.get("M3_direct", {}), d.get("M4_controlled", {})
    rows.append({
        "model": k,
        "M3_obj_max_patched_LD": m3.get("object", {}).get("max_mean_abs_patched_LD"),
        "M3_obj_any_cross0": m3.get("object", {}).get("any_layer_mean_cross0"),
        "M3_subj_max_patched_LD": m3.get("subject", {}).get("max_mean_abs_patched_LD"),
        "M4_subj_honor_max": m4.get("subject", {}).get("honor_max_LD"),
        "M4_obj_honor_max": m4.get("object", {}).get("honor_max_LD"),
        "M4_obj_cells_cross0": f"{m4.get('object', {}).get('honor_cells_cross0')}/{m4.get('object', {}).get('n_cells')}",
        "M4_obj_per_lexeme": m4.get("object_per_lexeme_max_LD"),
    })

print(f"\n{'모델':10}{'M3주체최대':>11}{'M3객체최대':>11}{'M4주체최대':>11}{'M4객체최대':>11}{'객체0넘는셀':>11}")
for r in rows:
    print(f"{r['model']:10}{str(r['M3_subj_max_patched_LD']):>11}{str(r['M3_obj_max_patched_LD']):>11}"
          f"{str(r['M4_subj_honor_max']):>11}{str(r['M4_obj_honor_max']):>11}{str(r['M4_obj_cells_cross0']):>11}")
print("\n객체 어휘별 최대 조향 LD:")
for r in rows:
    print(f"  {r['model']}: {json.dumps(r['M4_obj_per_lexeme'], ensure_ascii=False)}")

(OUT / "compare_causal.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

if avail:
    fig, ax = plt.subplots(figsize=(max(6, 2.4 * len(avail)), 4.6))
    x = np.arange(len(avail)); w = 0.36
    subj = [data[k]["M4_controlled"]["subject"]["honor_max_LD"] for k in avail]
    obj = [data[k]["M4_controlled"]["object"]["honor_max_LD"] for k in avail]
    ax.bar(x - w / 2, subj, w, label="주체 -시- 최대 조향 LD", color="#2b6cb0")
    ax.bar(x + w / 2, obj, w, label="객체 보충법 최대 조향 LD", color="#c53030")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[k] for k in avail])
    ax.set_ylabel("전역 경어 벡터 조향 시 최대 logit-diff")
    ax.set_title("인과 M6: 조향으로 -시-는 강건 복구, 보충법은 취약(모델 공통)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(subj, obj)):
        ax.text(i - w / 2, a + 0.05, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + (0.05 if b >= 0 else -0.2), f"{b:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "compare_causal.png", dpi=150)
    print("저장: analysis/output/compare_causal.png")
