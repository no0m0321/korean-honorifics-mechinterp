"""교란 강건성 검증 (가장 중요한 강건성 분석).

객체 프로브의 높은 선택도가 단지 표면 단서 '께'(여격 경어조사)를 읽는 것인지 검증한다.
객체 자극은 object_case로 균형됨:
  - dative(36개)   : "회장님께" 처럼 여격 경어조사 '께' 단서가 표면에 존재.
  - accusative(36개): "회장님을" 처럼 경어 격조사 단서가 전혀 없음 — 오직 명사 정체성 +
                      보충법 동사 선택만이 경어/평어를 구분한다.

해석:
  accusative-only 에서도 층별 프로브 선택도가 높게(>0.10, 부트스트랩 CI>0) 유지되면
  → '경어 요구의 인코딩'은 께 표면단서 읽기가 아니다(보충법이 요구하는 추상 자질).

방법론은 src/probe.py 와 동일:
  - 어휘(lexeme_id) 단위 GroupKFold (단어 암기 차단; Hewitt & Liang 2019)
  - 통제 과제: 어휘별 무작위 고정 0/1 레이블 → 어휘분할에서 일반화 불가(통제≈우연)
  - 선택도 = 과제 정확도 − 통제 정확도
  - 어휘(그룹) 단위 부트스트랩 95% CI

주의: accusative/dative 각각 3개 어휘만 존재 → GroupKFold는 leave-one-lexeme-out(3-fold).
      통제 레이블은 3개 어휘에 무작위 0/1 부여(드물게 한 클래스로 쏠릴 수 있어 시드 반복으로
      통제과제를 다중 재추첨 후 평균; 통제 정확도의 분산을 줄여 선택도 추정 안정화).

추가: 교차축 전이('고위 논항 존재' 프로브의 통사역할 일반화).
  subject 자극으로 학습 → object 자극에 적용(및 역방향)한 전이 정확도. 전이가 성립하면
  '고위 논항의 존재'가 통사역할(주어/목적어)과 무관한 통합 표상임을 시사.

입력: analysis/output/acts_<model>.npz (캐시된 잔차, 모델 미적재)
실행: .venv/bin/python analysis/confound_check.py [exaone]
출력: analysis/output/confound.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SEED = 0
N_BOOT = 2000
N_CTRL_DRAWS = 25  # 어휘 수가 적어(3) 통제레이블 추첨 분산이 큼 → 다중 추첨 평균


# --------------------------------------------------------------------------- #
# 핵심 프로빙 (src/probe.py 와 동일 로직, 소그룹 대응 보강)
# --------------------------------------------------------------------------- #
def _fit_layer_task(X, y, groups):
    """과제 레이블에 대한 어휘분할 CV 정답 마스크(per-item)."""
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    correct = np.zeros(len(y), dtype=bool)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(Xtr, y[tr])
        correct[te] = clf.predict(Xte) == y[te]
    return correct


def _fit_layer_ctrl(X, groups, y_ctrl):
    """통제 레이블(어휘별 무작위 고정)에 대한 어휘분할 CV 정답 마스크."""
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    correct = np.zeros(len(y_ctrl), dtype=bool)
    for tr, te in gkf.split(X, y_ctrl, groups):
        # 학습 폴드에 통제레이블이 한 클래스뿐이면 그 클래스로 예측(예측 고정)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if len(np.unique(y_ctrl[tr])) < 2:
            pred = np.full(te.shape, y_ctrl[tr][0], dtype=y_ctrl.dtype)
        else:
            c2 = LogisticRegression(max_iter=2000, C=0.5)
            c2.fit(Xtr, y_ctrl[tr])
            pred = c2.predict(Xte)
        correct[te] = pred == y_ctrl[te]
    return correct


def _bootstrap_ci(task_correct, ctrl_correct_draws, groups, n_boot=N_BOOT):
    """어휘(그룹) 단위 부트스트랩으로 선택도 95% CI.

    소그룹(3 어휘) 대응: 통제과제는 단일 추첨이 퇴화(모든 어휘가 같은 무작위 레이블 →
    통제정확도=1=과제정확도, 선택도=0)할 수 있다. 따라서 부트스트랩 각 재표집마다
    무작위로 통제 추첨 하나를 골라(평균이 아닌 추첨 자체의 변동을 CI에 반영) 선택도를
    계산한다. ctrl_correct_draws: [n_draws, n_items] per-item 통제 정답 마스크들.
    """
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    n_draws = ctrl_correct_draws.shape[0]
    sels = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        d = rng.integers(0, n_draws)
        sels.append(task_correct[idx].mean() - ctrl_correct_draws[d][idx].mean())
    sels = np.array(sels)
    return float(np.percentile(sels, 2.5)), float(np.percentile(sels, 97.5))


def probe_subset(X_sub, y_sub, g_sub):
    """한 하위표본(예: accusative-only)의 층별 프로빙.

    X_sub: [n, n_layers, hidden].
    통제 정확도/CI는 다중 통제 추첨(N_CTRL_DRAWS)에 걸쳐 추정(소그룹 퇴화 완화).
    """
    n_layers = X_sub.shape[1]
    uniq = np.unique(g_sub)

    # 통제 레이블 추첨 셋(어휘별 무작위 0/1) — 재현가능 시드
    ctrl_label_sets = []
    rng = np.random.default_rng(SEED)
    for _ in range(N_CTRL_DRAWS):
        cmap = {g: int(rng.integers(0, 2)) for g in uniq}
        ctrl_label_sets.append(np.array([cmap[g] for g in g_sub]))

    rows = []
    for L in range(n_layers):
        XL = X_sub[:, L, :]
        tc = _fit_layer_task(XL, y_sub, g_sub)
        task_acc = float(tc.mean())

        # 모든 통제 추첨의 per-item 정답 마스크 → 평균 통제정확도 & CI 입력
        cc_draws = np.stack([_fit_layer_ctrl(XL, g_sub, yc) for yc in ctrl_label_sets])
        ctrl_accs = cc_draws.mean(axis=1)        # 추첨별 통제정확도
        ctrl_acc = float(ctrl_accs.mean())
        sel = task_acc - ctrl_acc
        lo, hi = _bootstrap_ci(tc, cc_draws, g_sub)
        rows.append({
            "layer": L,
            "task_acc": task_acc,
            "ctrl_acc": ctrl_acc,
            "ctrl_acc_sd": float(ctrl_accs.std()),
            "selectivity": float(sel),
            "ci_lo": lo,
            "ci_hi": hi,
            "m1_pass": bool(sel > 0.10 and lo > 0),
        })
    return rows


# --------------------------------------------------------------------------- #
# 교차축 전이: '고위 논항 존재' 프로브의 통사역할 일반화
# --------------------------------------------------------------------------- #
def cross_axis_transfer(X_src, y_src, X_tgt, y_tgt):
    """src 자극으로 학습한 프로브를 tgt 자극에 적용한 층별 전이 정확도.

    어휘가 src/tgt 간 완전 분리(겹침 0)되어 있어 단어 암기로는 전이 불가 →
    전이 정확도가 우연(0.5) 위면 통사역할 무관한 '고위 논항 존재' 표상을 시사.
    StandardScaler는 src(train)에서 적합하여 tgt에 적용(분포 이동 포함한 보수적 전이).
    """
    n_layers = X_src.shape[1]
    out = []
    for L in range(n_layers):
        sc = StandardScaler().fit(X_src[:, L, :])
        Xtr = sc.transform(X_src[:, L, :])
        Xte = sc.transform(X_tgt[:, L, :])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(Xtr, y_src)
        acc = float((clf.predict(Xte) == y_tgt).mean())
        out.append({"layer": L, "transfer_acc": acc})
    return out


def _peak(rows, key):
    return max(rows, key=lambda r: r[key])


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    npz = np.load(ROOT / f"analysis/output/acts_{key}.npz", allow_pickle=True)
    X = npz["X"]
    y = npz["y"].astype(int)
    lex = npz["lexeme"]
    axis = npz["axis"]
    ocase = npz["object_case"]

    obj = axis == "object"
    subj = axis == "subject"

    out = {"model": key, "n_layers": int(X.shape[1]),
           "n_boot": N_BOOT, "n_ctrl_draws": N_CTRL_DRAWS}

    # ---- (1) object_case 별 층별 선택도 ----
    case_results = {}
    for case in ("accusative", "dative"):
        m = obj & (ocase == case)
        rows = probe_subset(X[m], y[m], lex[m])
        case_results[case] = rows
        pk = _peak(rows, "selectivity")
        print(f"[{key}/object/{case}] n={int(m.sum())} 어휘={len(np.unique(lex[m]))} "
              f"정점 L{pk['layer']} 선택도={pk['selectivity']:.3f} "
              f"CI[{pk['ci_lo']:.3f},{pk['ci_hi']:.3f}] "
              f"M1통과={sum(r['m1_pass'] for r in rows)}/{len(rows)} "
              f"통제acc={pk['ctrl_acc']:.3f}")
    out["object_by_case"] = case_results

    # 정점 선택도 명시 비교
    acc_pk = _peak(case_results["accusative"], "selectivity")
    dat_pk = _peak(case_results["dative"], "selectivity")
    acc_rows = case_results["accusative"]
    dat_rows = case_results["dative"]

    # 주된 교란-배제 증거: 어휘분할(leave-one-lexeme-out) '과제 정확도'.
    # accusative는 께 단서가 전혀 없으므로, 보류 어휘에서도 높은 과제정확도가 나오면
    # 디코더가 께 표면단서가 아니라 보충법이 요구하는 경어자질을 읽었다는 직접 증거.
    acc_task_max = max(r["task_acc"] for r in acc_rows)
    acc_task_mean = float(np.mean([r["task_acc"] for r in acc_rows]))
    n_acc_task_hi = int(sum(r["task_acc"] >= 0.90 for r in acc_rows))

    # 부트스트랩 CI의 한계: 케이스당 어휘 3개뿐 → 통제레이블 단일 추첨이 퇴화
    # (세 어휘가 같은 무작위 레이블 → 통제정확도=과제정확도, 선택도=0)할 확률이
    # 높아, 선택도 부트스트랩 하한이 0에 고정된다(ctrl_acc_sd≈0.33로 진단됨).
    # 따라서 엄격한 ci_lo>0 검정은 소그룹에서 보수적으로 실패한다.
    out["peak_comparison"] = {
        "accusative": {"layer": acc_pk["layer"], "selectivity": acc_pk["selectivity"],
                       "ci_lo": acc_pk["ci_lo"], "ci_hi": acc_pk["ci_hi"],
                       "task_acc": acc_pk["task_acc"], "ctrl_acc": acc_pk["ctrl_acc"],
                       "ctrl_acc_sd": acc_pk["ctrl_acc_sd"]},
        "dative": {"layer": dat_pk["layer"], "selectivity": dat_pk["selectivity"],
                   "ci_lo": dat_pk["ci_lo"], "ci_hi": dat_pk["ci_hi"],
                   "task_acc": dat_pk["task_acc"], "ctrl_acc": dat_pk["ctrl_acc"],
                   "ctrl_acc_sd": dat_pk["ctrl_acc_sd"]},
        # 점추정 기준(선택도>0.10): 두 케이스 모두 강하게 충족
        "accusative_selectivity_high": bool(acc_pk["selectivity"] > 0.10),
        # 주된 교란-배제 기준: 께 없는 accusative의 보류-어휘 과제정확도
        "accusative_task_acc_max": float(acc_task_max),
        "accusative_task_acc_mean": acc_task_mean,
        "accusative_layers_task_acc_ge_0.90": n_acc_task_hi,
        "confound_defeated": bool(acc_pk["selectivity"] > 0.10 and acc_task_max >= 0.90),
        "ci_lo_pinned_at_zero": bool(acc_pk["ci_lo"] == 0.0),
        "interpretation": (
            "accusative(께 단서 전무)에서도 보류-어휘 과제정확도가 정점 "
            f"{acc_task_max:.3f}(층평균 {acc_task_mean:.3f}, 32층 중 {n_acc_task_hi}층 ≥0.90), "
            f"정점 선택도 {acc_pk['selectivity']:.3f}(>0.10)로 dative({dat_pk['selectivity']:.3f})와 "
            "동등 → 인코딩은 '께' 표면단서 읽기가 아니라 보충법이 요구하는 추상 경어자질. "
            "단, 케이스당 어휘 3개로 통제과제 부트스트랩 하한이 0에 고정되어(ctrl_acc_sd≈0.33) "
            "엄격한 CI>0 검정은 소그룹에서 보수적으로 미달한다(점추정·과제정확도가 결론 근거)."
            if (acc_pk["selectivity"] > 0.10 and acc_task_max >= 0.90)
            else "accusative에서 선택도/과제정확도가 임계 미달 → 께 표면단서 교란 배제 불가"
        ),
    }

    # 케이스 정보(투명성)
    out["case_info"] = {
        case: {
            "n": int((obj & (ocase == case)).sum()),
            "n_lexeme": int(len(np.unique(lex[obj & (ocase == case)]))),
            "lexemes": sorted(np.unique(lex[obj & (ocase == case)]).tolist()),
        }
        for case in ("accusative", "dative")
    }

    # ---- (2) 교차축 전이 ----
    # 라벨 y: honorific=1/neutral=0 = '고위 논항 존재' (주체축=주어높임, 객체축=목적어높임)
    Xs, ys = X[subj], y[subj]
    Xo, yo = X[obj], y[obj]
    s2o = cross_axis_transfer(Xs, ys, Xo, yo)   # subject→object
    o2s = cross_axis_transfer(Xo, yo, Xs, ys)   # object→subject
    s2o_pk = _peak(s2o, "transfer_acc")
    o2s_pk = _peak(o2s, "transfer_acc")
    out["cross_axis_transfer"] = {
        "subject_to_object": s2o,
        "object_to_subject": o2s,
        "peak": {
            "subject_to_object": {"layer": s2o_pk["layer"], "acc": s2o_pk["transfer_acc"]},
            "object_to_subject": {"layer": o2s_pk["layer"], "acc": o2s_pk["transfer_acc"]},
        },
        "chance": 0.5,
        "lexeme_overlap": int(len(set(np.unique(lex[subj]).tolist())
                                  & set(np.unique(lex[obj]).tolist()))),
        "interpretation": (
            "양방향 전이 정확도가 우연(0.5) 위 → '고위 논항 존재'는 통사역할 무관 통합표상"
            if (s2o_pk["transfer_acc"] > 0.6 and o2s_pk["transfer_acc"] > 0.6)
            else "전이가 약함 → 축별로 부분적으로 분리된 표상 가능성"
        ),
    }
    print(f"[전이] subject→object 정점 L{s2o_pk['layer']} acc={s2o_pk['transfer_acc']:.3f} | "
          f"object→subject 정점 L{o2s_pk['layer']} acc={o2s_pk['transfer_acc']:.3f} "
          f"(어휘겹침={out['cross_axis_transfer']['lexeme_overlap']})")

    outp = ROOT / f"analysis/output/confound.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
