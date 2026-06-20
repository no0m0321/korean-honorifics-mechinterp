"""M2 통계: 로짓 렌즈 추월(crossover) 비대칭의 통계적 검정.

데이터: analysis/output/lens_exaone.npz
  diff[264,32] : per-item per-layer logit_diff = logit(경어형 첫토큰) - logit(평형형 첫토큰)
  axis ('subject'/'object'), cond ('honorific'/'neutral'), lexeme (어휘 id)
  행 순서는 pairs.jsonl 과 동일, 무효행 없음(NaN 0).

분석(honorific 조건만):
 1) 항목별 최종 logit_diff(LD, 마지막 층)의 축별 검정.
    (a) 주체 최종 LD > 0,  (b) 객체 최종 LD < 0,  (c) 주체 > 객체.
    어휘(lexeme)를 군집으로 한 cluster-robust OLS(군집수가 적으므로 어휘 부트스트랩,
    및 어휘 평균 기반 비모수 검정으로 교차검증).
 2) 항목별 추월층(per-item crossover): diff[item,:]가 음->양으로 전환 후 끝까지 유지되는
    첫 층. 없으면 결측. 축별 추월있음/없음 비율, Fisher exact(추월×축).
    존재 항목의 추월층 분포(중앙값/IQR) 및 Mann-Whitney.
 3) 효과크기 = 주체 최종 LD 평균 - 객체 최종 LD 평균, 어휘 군집 부트스트랩 95% CI.

폴백 정책: 혼합효과/복잡 모형은 군집수가 적어(주체16/객체6) 불안정하므로,
cluster-robust OLS(어휘 군집)를 1차 추론으로, 어휘 군집 부트스트랩을 병행 보고.

실행: .venv/bin/python analysis/stats_m2.py
저장: analysis/output/stats_m2.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260620)
N_BOOT = 10000


def crossover_layer(curve):
    """음->양으로 처음 바뀌고 이후 끝까지 양으로 유지되는 첫 층. 없으면 None."""
    nL = len(curve)
    for L in range(nL):
        if curve[L] > 0 and all(curve[k] > 0 for k in range(L, nL)):
            return L
    return None


def cluster_bootstrap_mean(values, clusters, n=N_BOOT):
    """어휘 군집 부트스트랩: 군집을 재표집해 평균의 분포를 만든다."""
    uniq = np.unique(clusters)
    by = {c: values[clusters == c] for c in uniq}
    out = np.empty(n)
    for i in range(n):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by[c] for c in pick])
        out[i] = vals.mean()
    return out


def cluster_bootstrap_diff(v1, c1, v2, c2, n=N_BOOT):
    """두 그룹(주체/객체) 평균 차의 군집 부트스트랩 분포."""
    u1, u2 = np.unique(c1), np.unique(c2)
    by1 = {c: v1[c1 == c] for c in u1}
    by2 = {c: v2[c2 == c] for c in u2}
    out = np.empty(n)
    for i in range(n):
        p1 = RNG.choice(u1, size=len(u1), replace=True)
        p2 = RNG.choice(u2, size=len(u2), replace=True)
        m1 = np.concatenate([by1[c] for c in p1]).mean()
        m2 = np.concatenate([by2[c] for c in p2]).mean()
        out[i] = m1 - m2
    return out


def boot_ci(samples, alpha=0.05):
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def boot_p_twosided(samples, null=0.0):
    """부트스트랩 분포에서 null 값을 기준으로 한 양측 p (분포가 null을 넘는 비율 기반)."""
    p_greater = float(np.mean(samples <= null))
    p_less = float(np.mean(samples >= null))
    p = 2 * min(p_greater, p_less)
    return min(1.0, p)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def main():
    z = np.load(ROOT / "analysis/output/lens_exaone.npz", allow_pickle=True)
    diff, axis, cond, lexeme = z["diff"], z["axis"], z["cond"], z["lexeme"]
    nL = diff.shape[1]

    hon = cond == "honorific"
    final_ld = diff[:, -1]

    res = {
        "model": "exaone",
        "n_layers": int(nL),
        "measure": "final_logit_diff = logit(경어형 첫토큰) - logit(평형형 첫토큰) @ 마지막 층",
        "condition": "honorific only",
        "n_boot": N_BOOT,
        "seed": 20260620,
    }

    # 그룹 정의 (honorific)
    sm_mask = hon & (axis == "subject")
    ob_mask = hon & (axis == "object")
    subj_ld, subj_lex = final_ld[sm_mask], lexeme[sm_mask]
    obj_ld, obj_lex = final_ld[ob_mask], lexeme[ob_mask]

    res["counts"] = {
        "subject_n_items": int(sm_mask.sum()),
        "subject_n_lexemes": int(len(np.unique(subj_lex))),
        "object_n_items": int(ob_mask.sum()),
        "object_n_lexemes": int(len(np.unique(obj_lex))),
    }

    # ---------- (1) 축별 최종 LD 부호/대조 검정 ----------
    # 기술통계
    def descr(v):
        return {
            "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
            "frac_positive": float((v > 0).mean()), "n": int(len(v)),
        }

    res["test1_descriptives"] = {"subject": descr(subj_ld), "object": descr(obj_ld)}

    # (1a) 주체 최종 LD > 0 : 어휘 평균 기반 일표본 (어휘를 단위로) Wilcoxon + cluster-bootstrap
    subj_lex_means = np.array([subj_ld[subj_lex == c].mean() for c in np.unique(subj_lex)])
    obj_lex_means = np.array([obj_ld[obj_lex == c].mean() for c in np.unique(obj_lex)])

    # Wilcoxon signed-rank on lexeme means vs 0
    w_s = stats.wilcoxon(subj_lex_means, alternative="greater")
    w_o = stats.wilcoxon(obj_lex_means, alternative="less")
    # cluster bootstrap of the mean
    bs_subj = cluster_bootstrap_mean(subj_ld, subj_lex)
    bs_obj = cluster_bootstrap_mean(obj_ld, obj_lex)

    res["test1a_subject_gt0"] = {
        "lexeme_mean_mean": float(subj_lex_means.mean()),
        "wilcoxon_stat": float(w_s.statistic),
        "wilcoxon_p_greater": float(w_s.pvalue),
        "cluster_boot_mean": float(bs_subj.mean()),
        "cluster_boot_ci95": list(boot_ci(bs_subj)),
        "cluster_boot_p_vs0_twosided": boot_p_twosided(bs_subj),
        "frac_lexemes_positive": float((subj_lex_means > 0).mean()),
    }
    res["test1b_object_lt0"] = {
        "lexeme_mean_mean": float(obj_lex_means.mean()),
        "wilcoxon_stat": float(w_o.statistic),
        "wilcoxon_p_less": float(w_o.pvalue),
        "cluster_boot_mean": float(bs_obj.mean()),
        "cluster_boot_ci95": list(boot_ci(bs_obj)),
        "cluster_boot_p_vs0_twosided": boot_p_twosided(bs_obj),
        "frac_lexemes_negative": float((obj_lex_means < 0).mean()),
    }

    # (1c) 주체 > 객체 : cluster-robust OLS (lexeme 군집)
    df = pd.DataFrame({
        "ld": np.concatenate([subj_ld, obj_ld]),
        "axis": (["subject"] * len(subj_ld)) + (["object"] * len(obj_ld)),
        "lexeme": np.concatenate([subj_lex, obj_lex]),
    })
    # object를 기준(0), subject=1 더미
    df["is_subject"] = (df["axis"] == "subject").astype(int)
    ols = smf.ols("ld ~ is_subject", data=df)
    fit_robust = ols.fit(cov_type="cluster", cov_kwds={"groups": df["lexeme"]})

    res["test1c_subject_vs_object_OLS_cluster_robust"] = {
        "note": "OLS ld ~ is_subject, cluster-robust SE on lexeme (군집수=22). intercept=객체 평균, is_subject=주체-객체 차.",
        "intercept_object_mean": float(fit_robust.params["Intercept"]),
        "coef_subject_minus_object": float(fit_robust.params["is_subject"]),
        "se_cluster": float(fit_robust.bse["is_subject"]),
        "t": float(fit_robust.tvalues["is_subject"]),
        "p_value": float(fit_robust.pvalues["is_subject"]),
        "ci95": [float(x) for x in fit_robust.conf_int().loc["is_subject"].tolist()],
        "n_clusters": int(df["lexeme"].nunique()),
    }

    # 비모수 보강: 어휘 평균 Mann-Whitney (주체 vs 객체)
    mw = stats.mannwhitneyu(subj_lex_means, obj_lex_means, alternative="greater")
    res["test1c_lexememeans_mannwhitney"] = {
        "U": float(mw.statistic), "p_greater": float(mw.pvalue),
        "subject_lexeme_mean": float(subj_lex_means.mean()),
        "object_lexeme_mean": float(obj_lex_means.mean()),
    }

    # ---------- (2) 항목별 추월층 분포 ----------
    co_subj = np.array([crossover_layer(diff[i]) for i in np.where(sm_mask)[0]], dtype=object)
    co_obj = np.array([crossover_layer(diff[i]) for i in np.where(ob_mask)[0]], dtype=object)

    subj_has = np.array([c is not None for c in co_subj])
    obj_has = np.array([c is not None for c in co_obj])

    # Fisher exact: 추월있음/없음 × 축
    table = np.array([
        [int(subj_has.sum()), int((~subj_has).sum())],   # subject: has, no
        [int(obj_has.sum()), int((~obj_has).sum())],      # object: has, no
    ])
    odds, fisher_p = stats.fisher_exact(table, alternative="greater")
    chi2, chi2_p, dof, _ = stats.chi2_contingency(table, correction=False)

    subj_co_vals = np.array([c for c in co_subj if c is not None], dtype=float)
    obj_co_vals = np.array([c for c in co_obj if c is not None], dtype=float)

    res["test2_crossover"] = {
        "definition": "per-item crossover = diff[item,:]가 음->양 전환 후 끝까지 유지되는 첫 층; 없으면 결측",
        "subject": {
            "n_items": int(sm_mask.sum()),
            "n_has_crossover": int(subj_has.sum()),
            "frac_has_crossover": float(subj_has.mean()),
            "crossover_layer_median": (float(np.median(subj_co_vals)) if len(subj_co_vals) else None),
            "crossover_layer_iqr": ([float(np.percentile(subj_co_vals, 25)),
                                     float(np.percentile(subj_co_vals, 75))] if len(subj_co_vals) else None),
            "crossover_layer_min_max": ([float(subj_co_vals.min()), float(subj_co_vals.max())]
                                        if len(subj_co_vals) else None),
        },
        "object": {
            "n_items": int(ob_mask.sum()),
            "n_has_crossover": int(obj_has.sum()),
            "frac_has_crossover": float(obj_has.mean()),
            "crossover_layer_median": (float(np.median(obj_co_vals)) if len(obj_co_vals) else None),
            "crossover_layer_iqr": ([float(np.percentile(obj_co_vals, 25)),
                                     float(np.percentile(obj_co_vals, 75))] if len(obj_co_vals) else None),
            "crossover_layer_min_max": ([float(obj_co_vals.min()), float(obj_co_vals.max())]
                                        if len(obj_co_vals) else None),
        },
        "contingency_table_[has,no]_x_[subject,object]": table.tolist(),
        "fisher_exact_odds_ratio": float(odds),
        "fisher_exact_p_greater": float(fisher_p),
        "chi2": float(chi2), "chi2_dof": int(dof), "chi2_p": float(chi2_p),
    }

    # 추월층 존재 항목들의 층 분포 비교(있으면)
    if len(subj_co_vals) and len(obj_co_vals):
        mw2 = stats.mannwhitneyu(subj_co_vals, obj_co_vals, alternative="two-sided")
        res["test2_crossover"]["layer_distribution_mannwhitney"] = {
            "U": float(mw2.statistic), "p_twosided": float(mw2.pvalue),
            "note": "추월층이 존재하는 항목만 비교",
        }
    else:
        res["test2_crossover"]["layer_distribution_mannwhitney"] = {
            "note": "한 축의 추월 항목 수가 0이라 비교 불가",
        }

    # ---------- (3) 효과크기 + CI ----------
    bs_diff = cluster_bootstrap_diff(subj_ld, subj_lex, obj_ld, obj_lex)
    res["test3_effect_size"] = {
        "subject_minus_object_final_LD": float(subj_ld.mean() - obj_ld.mean()),
        "cluster_boot_mean_diff": float(bs_diff.mean()),
        "cluster_boot_ci95": list(boot_ci(bs_diff)),
        "cluster_boot_p_vs0_twosided": boot_p_twosided(bs_diff),
        "cohens_d_item_level": cohens_d(subj_ld, obj_ld),
        "cohens_d_lexeme_level": cohens_d(subj_lex_means, obj_lex_means),
        "note": "효과크기는 주체-객체 최종 logit_diff 차. CI는 어휘 군집 부트스트랩.",
    }

    outp = ROOT / "analysis/output/stats_m2.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("저장:", outp)


if __name__ == "__main__":
    main()
