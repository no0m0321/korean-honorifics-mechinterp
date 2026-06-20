"""세 모델 M6 종합 비교 (1·2단계 통합).

가설 M6: 인코딩(M1)은 EXAONE에서 더 강하고 이른 층에 형성되나, 검색 병목(M2 추월층
부재)은 세 계열 공통. 모델별 프로빙 정점 선택도·정점층(상대깊이)과 로짓 렌즈 추월층·최종
logit_diff를 한 표·그림으로 통합한다.

실행: .venv/bin/python analysis/compare_models.py
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
for cand in ["AppleGothic", "AppleSDGothicNeo", "NanumGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

KEYS = ["exaone", "qwen", "llama"]
NAMES = {"exaone": "EXAONE(한국어)", "qwen": "Qwen(다국어)", "llama": "Llama(영어)"}


def peak(rows):
    r = max(rows, key=lambda x: x["selectivity"])
    return r["selectivity"], r["layer"], len(rows)


def main():
    avail = [k for k in KEYS if (OUT / f"probe_{k}.json").exists()]
    print(f"분석 가능한 모델: {avail}")
    table = []
    for k in avail:
        probe = json.loads((OUT / f"probe_{k}.json").read_text())
        ls = OUT / f"lens_{k}_summary.json"
        lens = json.loads(ls.read_text()) if ls.exists() else {}
        ssel, slay, nL = peak(probe["subject"])
        osel, olay, _ = peak(probe["object"])
        row = {
            "model": k,
            "n_layers": nL,
            "subj_peak_sel": round(ssel, 3), "subj_peak_depth": round(slay / nL, 2),
            "obj_peak_sel": round(osel, 3), "obj_peak_depth": round(olay / nL, 2),
            "subj_crossover": lens.get("subject_honorific", {}).get("crossover_layer"),
            "subj_final_ld": lens.get("subject_honorific", {}).get("final_logit_diff"),
            "obj_crossover": lens.get("object_honorific", {}).get("crossover_layer"),
            "obj_final_ld": lens.get("object_honorific", {}).get("final_logit_diff"),
        }
        table.append(row)

    # 표 출력
    print(f"\n{'모델':14}{'주체선택도':>10}{'객체선택도':>10}{'주체추월':>10}{'객체추월':>10}"
          f"{'주체최종LD':>11}{'객체최종LD':>11}")
    for r in table:
        print(f"{NAMES.get(r['model'], r['model']):14}"
              f"{r['subj_peak_sel']:>10}{r['obj_peak_sel']:>10}"
              f"{str(r['subj_crossover']):>10}{str(r['obj_crossover']):>10}"
              f"{str(r['subj_final_ld']):>11}{str(r['obj_final_ld']):>11}")

    (OUT / "compare_models.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(avail) >= 2:
        # 그림: 모델별 선택도 곡선 (subject vs object)
        fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 4.2), sharey=True)
        if len(avail) == 1:
            axes = [axes]
        for ax, k in zip(axes, avail):
            probe = json.loads((OUT / f"probe_{k}.json").read_text())
            for axis, col, lab in [("subject", "#2b6cb0", "주체 -시-"),
                                   ("object", "#c53030", "객체 보충법")]:
                rows = probe[axis]
                ax.plot([r["layer"] for r in rows], [r["selectivity"] for r in rows],
                        color=col, label=lab)
            ax.axhline(0.10, ls="--", color="gray", lw=1)
            ax.set_title(NAMES.get(k, k), fontsize=10)
            ax.set_xlabel("층"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("프로브 선택도")
        axes[0].legend(fontsize=8)
        fig.suptitle("M6: 모델 계열별 경어 요구 인코딩 (1단계)", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / "compare_probe.png", dpi=150)
        print("저장: analysis/output/compare_probe.png")


if __name__ == "__main__":
    main()
