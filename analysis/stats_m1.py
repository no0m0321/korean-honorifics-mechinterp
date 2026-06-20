"""M1 (인코딩) 사전등록 혼합효과 통계.

연구 질문(RQ1): 주체높임 -시-(굴절)와 객체높임 보충법(드리다) 모두에서 '경어 요구(고위
논항의 존재)'가 동사 생성 직전 잔차흐름에 인코딩되는가? 만약 객체축에서도(또는 그 이상으로)
인코딩되면, 보충법 실패의 책임은 인코딩이 아니라 후기 검색에 있다는 1차 증거가 된다.

설계(사전등록):
  1) 각 축(subject/object)에서, probe_exaone.json 의 정점 층(축별 고정 L)을 사용.
     해당 층 잔차 X[:,L,:]로 GroupKFold(어휘 lexeme 단위, 5분할) 로지스틱회귀(StandardScaler,
     C=0.5; src.probe 와 동일) → 항목별 정답여부(과제). 동시에 통제과제(어휘별 무작위 고정
     레이블)도 같은 분할/같은 분류기로 → 항목별 정답여부(통제).
     선택도 = 과제정확도 − 통제정확도. (Hewitt & Liang 2019 통제프로브)
  2) 두 축을 한 데이터프레임으로 합쳐, 항목별 과제정답여부 ~ axis(객체=1/주체=0) 를
     혼합효과 로지스틱(BinomialBayesMixedGLM, 무선절편=lexeme + frame)으로 적합.
     axis 사후평균·95% 신용구간 보고. 수렴/적합 문제 시 GEE(Binomial, cluster=frame),
     그래도 안 되면 cluster-robust Logit(cov=cluster frame) 으로 폴백하고 무엇을 했는지 보고.
  3) 핵심 검정:
     (a) 각 축 선택도 > 0  — 어휘재표집 부트스트랩 95% CI 가 0 초과.
     (b) 객체 선택도 > 주체 선택도 — 어휘재표집 부트스트랩으로 (객체−주체) 차이 분포의
         95% CI 와 단측 p(>0) 보고.

입력: analysis/output/acts_exaone.npz, analysis/output/probe_exaone.json
출력: analysis/output/stats_m1.json
실행: .venv/bin/python analysis/stats_m1.py [exaone]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 0
N_BOOT = 2000


# ----------------------------------------------------------------------------
# 1) 항목별 정답여부 산출 (src.probe 와 동일한 어휘분할 CV / 통제프로브)
# ----------------------------------------------------------------------------
def fit_layer_correct(X, y, groups, y_ctrl, n_splits=5):
    """한 층(이미 슬라이스된 [n,hidden])에서 어휘분할 CV → 항목별 정답여부(과제·통제)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    task_correct = np.zeros(len(y), dtype=bool)
    ctrl_correct = np.zeros(len(y), dtype=bool)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(Xtr, y[tr])
        task_correct[te] = clf.predict(Xte) == y[te]
        # 통제: 어휘분할로 인해 훈련폴드의 통제레이블이 한 클래스만일 수 있음
        # (객체축 어휘 6개 → 무작위 고정레이블이 전부 동일할 가능성). 이때
        # 로지스틱은 적합 불가이므로 그 상수 클래스를 예측하는 퇴화분류기로 처리.
        cls = np.unique(y_ctrl[tr])
        if len(cls) < 2:
            pred = np.full(te.shape, cls[0])
        else:
            c2 = LogisticRegression(max_iter=2000, C=0.5)
            c2.fit(Xtr, y_ctrl[tr])
            pred = c2.predict(Xte)
        ctrl_correct[te] = pred == y_ctrl[te]
    return task_correct, ctrl_correct


def control_labels(groups, rng):
    """어휘별 무작위 0/1 고정 레이블."""
    uniq = np.unique(groups)
    ctrl_map = {g: int(rng.integers(0, 2)) for g in uniq}
    return np.array([ctrl_map[g] for g in groups])


def peak_layers(key):
    """probe JSON 에서 축별 정점 선택도 층."""
    d = json.loads((ROOT / f"analysis/output/probe_{key}.json").read_text())
    peaks = {}
    for ax in ("subject", "object"):
        rows = d[ax]
        peak = max(rows, key=lambda r: r["selectivity"])
        peaks[ax] = int(peak["layer"])
    return peaks


# ----------------------------------------------------------------------------
# 2) 부트스트랩 (어휘 단위 재표집)
# ----------------------------------------------------------------------------
def bootstrap_selectivity(task_c, ctrl_c, groups, n_boot=N_BOOT, seed=SEED):
    """한 축 선택도(과제−통제)의 어휘재표집 분포 → 표본 배열 반환."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_g[g] for g in pick])
        out[b] = task_c[idx].mean() - ctrl_c[idx].mean()
    return out


def bootstrap_diff(task_o, ctrl_o, grp_o, task_s, ctrl_s, grp_s,
                   n_boot=N_BOOT, seed=SEED):
    """(객체 선택도 − 주체 선택도) 차이의 어휘재표집 분포. 두 축 독립 재표집."""
    rng = np.random.default_rng(seed)
    uo, us = np.unique(grp_o), np.unique(grp_s)
    bo = {g: np.where(grp_o == g)[0] for g in uo}
    bs = {g: np.where(grp_s == g)[0] for g in us}
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        po = rng.choice(uo, size=len(uo), replace=True)
        io = np.concatenate([bo[g] for g in po])
        ps = rng.choice(us, size=len(us), replace=True)
        isx = np.concatenate([bs[g] for g in ps])
        sel_o = task_o[io].mean() - ctrl_o[io].mean()
        sel_s = task_s[isx].mean() - ctrl_s[isx].mean()
        diffs[b] = sel_o - sel_s
    return diffs


def ci_p(samples, value=0.0):
    """양측 2.5/97.5 백분위 CI 와 단측 p(samples<=value) (>0 우월성 검정용)."""
    lo, hi = np.percentile(samples, [2.5, 97.5])
    # 단측 p: H0 차이<=0 에 대한 부트스트랩 p ≈ 비율(표본 <= value)
    p_gt = float((samples <= value).mean())
    return float(lo), float(hi), p_gt


# ----------------------------------------------------------------------------
# 3) 혼합효과 로지스틱 (BinomialBayesMixedGLM) + 폴백 (GEE, cluster-robust)
# ----------------------------------------------------------------------------
def fit_bayes_mixed(df):
    """과제정답 ~ axis_obj + (1|lexeme) + (1|frame). 사후평균·95% 신용구간."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    # vc: 무선절편 두 개 (lexeme, frame)
    vc = {"lexeme": "0 + C(lexeme)", "frame": "0 + C(frame)"}
    md = BinomialBayesMixedGLM.from_formula(
        "correct ~ axis_obj", vc, df, vcp_p=3.0, fe_p=3.0)
    res = md.fit_vb()  # 변분 베이즈 (수렴 안정적)
    # 고정효과: fe_mean/fe_sd 순서는 exog_names
    names = list(res.model.exog_names)
    idx = names.index("axis_obj")
    mean = float(res.fe_mean[idx])
    sd = float(res.fe_sd[idx])
    lo = mean - 1.96 * sd
    hi = mean + 1.96 * sd
    return {
        "method": "BinomialBayesMixedGLM(fit_vb)",
        "random_effects": ["lexeme", "frame"],
        "axis_obj_coef": mean,
        "axis_obj_sd": sd,
        "axis_obj_ci95": [float(lo), float(hi)],
        "axis_obj_or": float(np.exp(mean)),
        "axis_obj_or_ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "intercept": float(res.fe_mean[names.index("Intercept")]),
        "converged": True,
    }


def fit_gee(df):
    """폴백 1: GEE(Binomial, exchangeable, cluster=frame)."""
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    md = GEE.from_formula("correct ~ axis_obj", groups="frame",
                          data=df, family=Binomial(), cov_struct=Exchangeable())
    res = md.fit()
    coef = float(res.params["axis_obj"])
    ci = res.conf_int().loc["axis_obj"].values
    return {
        "method": "GEE(Binomial, Exchangeable, cluster=frame)",
        "axis_obj_coef": coef,
        "axis_obj_se": float(res.bse["axis_obj"]),
        "axis_obj_z": float(res.tvalues["axis_obj"]),
        "axis_obj_p": float(res.pvalues["axis_obj"]),
        "axis_obj_ci95": [float(ci[0]), float(ci[1])],
        "axis_obj_or": float(np.exp(coef)),
        "axis_obj_or_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
        "intercept": float(res.params["Intercept"]),
        "converged": bool(res.converged) if hasattr(res, "converged") else True,
    }


def fit_cluster_robust_logit(df):
    """폴백 2: cluster-robust Logit (cov=cluster, groups=frame)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    res = smf.logit("correct ~ axis_obj", data=df).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["frame"]})
    coef = float(res.params["axis_obj"])
    ci = res.conf_int().loc["axis_obj"].values
    return {
        "method": "cluster-robust Logit (cluster=frame)",
        "axis_obj_coef": coef,
        "axis_obj_se": float(res.bse["axis_obj"]),
        "axis_obj_z": float(res.tvalues["axis_obj"]),
        "axis_obj_p": float(res.pvalues["axis_obj"]),
        "axis_obj_ci95": [float(ci[0]), float(ci[1])],
        "axis_obj_or": float(np.exp(coef)),
        "axis_obj_or_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
        "intercept": float(res.params["Intercept"]),
        "converged": True,
    }


def run_mixed(df):
    """베이즈혼합 → GEE → cluster-robust 순으로 시도, 폴백 사유 기록."""
    fallbacks = []
    # 항목별 과제정답여부가 한 축에서 전부 1(완전분리)이면 로지스틱이 불안정.
    # 그래도 베이즈혼합은 사전분포로 정규화되므로 우선 시도.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_bayes_mixed(df)
        # 비정상값 점검
        if not np.isfinite(res["axis_obj_coef"]) or res["axis_obj_sd"] > 50:
            raise RuntimeError("비정상 사후(계수 비유한 또는 sd 과대)")
        res["fallback_log"] = fallbacks
        return res
    except Exception as e:
        fallbacks.append(f"BinomialBayesMixedGLM 실패/불안정 → {type(e).__name__}: {e}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_gee(df)
        res["fallback_log"] = fallbacks
        return res
    except Exception as e:
        fallbacks.append(f"GEE 실패 → {type(e).__name__}: {e}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_cluster_robust_logit(df)
    res["fallback_log"] = fallbacks
    return res


# ----------------------------------------------------------------------------
def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    npz = np.load(ROOT / f"analysis/output/acts_{key}.npz", allow_pickle=True)
    X = npz["X"]              # [n_items, n_layers, hidden]
    y = npz["y"]             # honorific=1 / neutral=0
    lexeme = npz["lexeme"]
    axis = npz["axis"]
    frame = npz["frame"]

    peaks = peak_layers(key)  # {'subject':L, 'object':L}

    per_axis = {}
    item_records = []  # 혼합모형용 행
    rng_master = np.random.default_rng(SEED)

    for ax in ("subject", "object"):
        mask = axis == ax
        L = peaks[ax]
        Xa = X[mask][:, L, :]
        ya = y[mask]
        ga = lexeme[mask]
        fa = frame[mask]
        # 통제 레이블(어휘별 무작위 고정) — 축별 독립 시드로 재현
        rng = np.random.default_rng(SEED + (0 if ax == "subject" else 1))
        yc = control_labels(ga, rng)

        task_c, ctrl_c = fit_layer_correct(Xa, ya, ga, yc)
        task_acc = float(task_c.mean())
        ctrl_acc = float(ctrl_c.mean())
        sel = task_acc - ctrl_acc

        boot = bootstrap_selectivity(task_c, ctrl_c, ga,
                                     seed=SEED + (10 if ax == "subject" else 20))
        lo, hi, p_le0 = ci_p(boot, 0.0)

        per_axis[ax] = {
            "layer": L,
            "n_items": int(mask.sum()),
            "n_lexeme": int(len(np.unique(ga))),
            "n_frame": int(len(np.unique(fa))),
            "task_acc": task_acc,
            "ctrl_acc": ctrl_acc,
            "selectivity": float(sel),
            "selectivity_ci95": [lo, hi],
            "selectivity_gt0": bool(lo > 0),
            "boot_p_selectivity_le0": p_le0,
            # 부트스트랩 표본은 (b) 차이검정에서 재사용하지 않음(독립 재표집)
            "_task_correct": task_c,
            "_ctrl_correct": ctrl_c,
            "_groups": ga,
        }
        for i in range(len(ya)):
            item_records.append({
                "correct": int(task_c[i]),
                "axis_obj": 1 if ax == "object" else 0,
                "axis": ax,
                "lexeme": str(ga[i]),
                "frame": str(fa[i]),
            })

    df = pd.DataFrame(item_records)

    # (2) 혼합효과 로지스틱: 과제정답 ~ axis(객체=1)
    mixed = run_mixed(df)

    # (3b) 객체 선택도 > 주체 선택도 차이 부트스트랩
    o, s = per_axis["object"], per_axis["subject"]
    diff_boot = bootstrap_diff(
        o["_task_correct"], o["_ctrl_correct"], o["_groups"],
        s["_task_correct"], s["_ctrl_correct"], s["_groups"],
        seed=SEED + 99)
    d_lo, d_hi, d_p_le0 = ci_p(diff_boot, 0.0)
    diff_point = o["selectivity"] - s["selectivity"]

    # 결과 정리(내부 배열 제거)
    for ax in per_axis:
        for k in ("_task_correct", "_ctrl_correct", "_groups"):
            per_axis[ax].pop(k)

    out = {
        "model": key,
        "measure_position": "동사 생성 직전 잔차 (probe 정점 층 고정)",
        "peak_layers": peaks,
        "n_boot": N_BOOT,
        "seed": SEED,
        "cv": "GroupKFold(lexeme, 5), StandardScaler, LogisticRegression(C=0.5)",
        "control_task": "어휘별 무작위 0/1 고정 레이블 (Hewitt & Liang 2019)",
        "per_axis": per_axis,
        "mixed_effects": mixed,
        "selectivity_difference_object_minus_subject": {
            "point": float(diff_point),
            "boot_ci95": [d_lo, d_hi],
            "boot_p_diff_le0": d_p_le0,
            "object_gt_subject_sig": bool(d_lo > 0),
        },
        "core_tests": {
            "a_subject_selectivity_gt0": bool(per_axis["subject"]["selectivity_ci95"][0] > 0),
            "a_object_selectivity_gt0": bool(per_axis["object"]["selectivity_ci95"][0] > 0),
            "b_object_gt_subject": bool(d_lo > 0),
        },
    }

    outp = ROOT / f"analysis/output/stats_m1.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 요약
    print("=== M1 인코딩 통계 (%s) ===" % key)
    for ax in ("subject", "object"):
        a = per_axis[ax]
        print(f"[{ax}] L{a['layer']} task={a['task_acc']:.3f} ctrl={a['ctrl_acc']:.3f} "
              f"sel={a['selectivity']:.3f} CI[{a['selectivity_ci95'][0]:.3f},"
              f"{a['selectivity_ci95'][1]:.3f}] >0={a['selectivity_gt0']}")
    print(f"[mixed] {mixed['method']} axis_obj coef={mixed['axis_obj_coef']:.3f} "
          f"CI95={['%.3f'%v for v in mixed['axis_obj_ci95']]} "
          f"OR={mixed['axis_obj_or']:.3f}")
    if mixed.get("fallback_log"):
        print("  폴백사유:", mixed["fallback_log"])
    print(f"[diff obj-subj] point={diff_point:.3f} "
          f"CI[{d_lo:.3f},{d_hi:.3f}] p(<=0)={d_p_le0:.4f} sig={d_lo>0}")
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
