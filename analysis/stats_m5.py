"""M5: 빈도 통제 후에도 굴절(-시-)-보충법(드리다) 해리가 잔존하는가.

연구 질문: 객체높임 보충법의 '실패'(최종 logit_diff가 낮음)가 단지 경어형 어휘의
저빈도(검색 실패의 사소한 설명) 때문인가, 아니면 빈도를 통제해도 굴절/보충법
'기제(axis)' 차이가 남는가(=빈도+기제 혼합 또는 순수 기제).

응답변수: lens_exaone.npz honorific 항목의 마지막 층(31) per-item logit_diff
          (= logit(경어형 첫토큰) - logit(평형형 첫토큰)). 양수=경어형 우세.

공변량(빈도):
  - 객체(보충법): verb_freq.json object_suppletive_pairs 의 log_freq_ratio
      = log(pln_freq/hon_freq) 근사. 클수록 경어형이 평형형보다 희귀(=검색 불리).
  - 주체(-시-): 생성적 굴절이라 보충쌍 빈도비가 정의되지 않음. 따라서
      (a) 결합모형에서는 객체 내부에서만 정의되는 log_freq_ratio 를 중심화 후
          주체엔 0(=객체평균)으로 두어 axis 더미가 '객체평균빈도에서의 축 격차'를
          잡도록 함(방법 명시). 추가로
      (b) subject_stems 평형형 토큰빈도로 만든 log-빈도 프록시로 빈도대 매칭
          부분집합 비교(robustness)를 수행. 두 빈도 척도는 종류가 달라 직접
          합치지 않고 별도 보고.

통계: 항목/어휘를 군집으로. 1차는 statsmodels 혼합효과(어휘 random intercept).
      수렴 실패 시 cluster-robust OLS(어휘 군집)로 폴백하고 무엇을 했는지 보고.

실행: .venv/bin/python analysis/stats_m5.py
출력: analysis/output/stats_m5.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
FINAL_LAYER = -1  # 마지막 층(=31)


def load_data():
    z = np.load(ROOT / "analysis/output/lens_exaone.npz", allow_pickle=True)
    diff, axis, cond, lex = z["diff"], z["axis"], z["cond"], z["lexeme"]
    rows = [json.loads(l) for l in open(ROOT / "data/stimuli/pairs.jsonl")]
    # 행 정렬 검증(메모리상 동일 순서)
    assert all(z["axis"][i] == rows[i]["axis"] for i in range(len(rows)))
    assert all(z["lexeme"][i] == rows[i]["lexeme_id"] for i in range(len(rows)))

    vf = json.load(open(ROOT / "data/freq/verb_freq.json"))
    # 객체 보충쌍: pln_stem -> log_freq_ratio. verb_lemma 매핑.
    # verb_lemma -> pln_stem 매핑(어간):
    lemma2stem = {
        "주다": "주", "말하다": "말하", "묻다": "묻",
        "보다": "보", "만나다": "만나", "데리고 가다": "데리",
    }
    stem2ratio = {p["pln_stem"]: p["log_freq_ratio"]
                  for p in vf["object_suppletive_pairs"]}
    # 단, '보→뵙'과 '만나→뵙' 모두 pln_stem 가 유일하므로 stem 으로 매칭 가능.
    obj_ratio = {lemma: stem2ratio[lemma2stem[lemma]] for lemma in lemma2stem}

    subj_stems = vf["subject_stems"]  # 평형형 토큰빈도(주체 프록시용)

    recs = []
    final = diff[:, FINAL_LAYER]
    for i, r in enumerate(rows):
        if r["cond"] != "honorific":
            continue
        rec = dict(
            item_id=r["item_id"], frame_id=r["frame_id"], axis=r["axis"],
            lexeme=r["lexeme_id"], verb_lemma=r["verb_lemma"],
            final_logit_diff=float(final[i]),
        )
        if r["axis"] == "object":
            rec["log_freq_ratio"] = float(obj_ratio[r["verb_lemma"]])
            rec["plain_freq"] = None
        else:  # subject
            rec["log_freq_ratio"] = np.nan  # 보충빈도비 정의 안 됨
            stem = r["verb_lemma"][:-1]  # '다' 제거
            rec["plain_freq"] = subj_stems.get(stem, None)
        recs.append(rec)
    return pd.DataFrame(recs)


def fit_cluster_ols(df, formula, cluster_col):
    """cluster-robust OLS. (groups, p, params, r2 반환)"""
    m = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    return m


def try_mixedlm(df, formula, group_col):
    """혼합효과(어휘 random intercept). 수렴 실패시 None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 수렴경고를 실패로 취급
            m = smf.mixedlm(formula, df, groups=df[group_col]).fit(reml=True)
        if not m.converged:
            return None, "not_converged"
        return m, "ok"
    except Exception as e:  # noqa
        return None, f"failed:{type(e).__name__}"


def main():
    df = load_data()
    out = {"model": "exaone", "response": "final(layer31) per-item logit_diff (honorific)",
           "n_total_honorific": int(len(df)),
           "n_subject": int((df.axis == "subject").sum()),
           "n_object": int((df.axis == "object").sum())}

    # 기술통계
    obj = df[df.axis == "object"].copy()
    subj = df[df.axis == "subject"].copy()
    out["descriptives"] = {
        "object_final_logit_diff_mean": round(float(obj.final_logit_diff.mean()), 4),
        "object_final_logit_diff_sd": round(float(obj.final_logit_diff.std(ddof=1)), 4),
        "subject_final_logit_diff_mean": round(float(subj.final_logit_diff.mean()), 4),
        "subject_final_logit_diff_sd": round(float(subj.final_logit_diff.std(ddof=1)), 4),
        "raw_axis_gap_subj_minus_obj": round(
            float(subj.final_logit_diff.mean() - obj.final_logit_diff.mean()), 4),
        "n_object_lexemes": int(obj.lexeme.nunique()),
        "object_lexeme_means": {
            lx: round(float(g.final_logit_diff.mean()), 3)
            for lx, g in obj.groupby("verb_lemma")},
        "object_log_freq_ratio_by_lexeme": {
            lx: float(g.log_freq_ratio.iloc[0])
            for lx, g in obj.groupby("verb_lemma")},
    }

    # ============================================================
    # 과제 2: 객체에서 final_logit_diff ~ log_freq_ratio (어휘 군집 robust)
    #   빈도가 보충법 실패를 얼마나 설명하는가(R²/계수).
    # ============================================================
    m_obj = fit_cluster_ols(obj, "final_logit_diff ~ log_freq_ratio", "lexeme")
    # 어휘 6개뿐 → 군집 robust SE는 소표본(군집 6) 한계. 보고에 명시.
    out["task2_object_freq_regression"] = {
        "formula": "final_logit_diff ~ log_freq_ratio  (cluster-robust by lexeme)",
        "n_obs": int(m_obj.nobs),
        "n_clusters_lexeme": int(obj.lexeme.nunique()),
        "intercept": round(float(m_obj.params["Intercept"]), 4),
        "slope_log_freq_ratio": round(float(m_obj.params["log_freq_ratio"]), 4),
        "slope_se": round(float(m_obj.bse["log_freq_ratio"]), 4),
        "slope_p": round(float(m_obj.pvalues["log_freq_ratio"]), 4),
        "r_squared": round(float(m_obj.rsquared), 4),
        "note": ("음의 기울기면 경어형이 희귀할수록(고 log_freq_ratio) 경어 우세가 "
                 "낮음=빈도가 실패에 기여. 군집 6개라 robust SE는 소표본 한계."),
    }
    # 보조: 어휘평균 수준 상관(어휘 6점) — 빈도가 설명하는 분산 상한
    lex_means = obj.groupby("verb_lemma").agg(
        y=("final_logit_diff", "mean"), x=("log_freq_ratio", "first")).reset_index()
    if len(lex_means) >= 3:
        r = np.corrcoef(lex_means.x, lex_means.y)[0, 1]
        out["task2_object_freq_regression"]["lexeme_level_pearson_r"] = round(float(r), 4)
        out["task2_object_freq_regression"]["lexeme_level_r_squared"] = round(float(r**2), 4)
        out["task2_object_freq_regression"]["lexeme_level_n"] = int(len(lex_means))

    # ============================================================
    # 과제 3: 빈도 통제 후에도 객체(보충) < 주체(굴절) 가 유의한가.
    #   결합모형: final_logit_diff ~ axis + log_freq_proxy
    #   빈도비(log_freq_ratio)는 객체 내부에서만 정의 → 중심화 후 주체=0(객체평균).
    #   axis 더미 = 객체평균빈도에서의 축 격차. 어휘 random intercept(혼합),
    #   실패 시 어휘 군집 robust OLS 폴백.
    # ============================================================
    df3 = df.copy()
    obj_mean_ratio = float(obj.log_freq_ratio.mean())
    # 중심화 빈도공변량: 객체는 (ratio - 객체평균), 주체는 0
    df3["freq_cov"] = np.where(
        df3.axis == "object", df3.log_freq_ratio - obj_mean_ratio, 0.0)
    df3["axis_object"] = (df3.axis == "object").astype(int)  # 1=객체(보충), 0=주체(굴절)

    formula3 = "final_logit_diff ~ axis_object + freq_cov"
    mm, mm_status = try_mixedlm(df3, formula3, "lexeme")
    method3 = None
    if mm is not None:
        method3 = "mixedlm (lexeme random intercept)"
        res3 = {
            "method": method3, "converged": True,
            "axis_object_coef": round(float(mm.params["axis_object"]), 4),
            "axis_object_se": round(float(mm.bse["axis_object"]), 4),
            "axis_object_z": round(float(mm.tvalues["axis_object"]), 4),
            "axis_object_p": float(mm.pvalues["axis_object"]),
            "freq_cov_coef": round(float(mm.params["freq_cov"]), 4),
            "freq_cov_p": float(mm.pvalues["freq_cov"]),
        }
    else:
        # 폴백: 어휘 군집 robust OLS
        m3 = fit_cluster_ols(df3, formula3, "lexeme")
        method3 = f"cluster-robust OLS by lexeme (mixedlm {mm_status})"
        res3 = {
            "method": method3, "converged": False, "mixedlm_status": mm_status,
            "axis_object_coef": round(float(m3.params["axis_object"]), 4),
            "axis_object_se": round(float(m3.bse["axis_object"]), 4),
            "axis_object_t": round(float(m3.tvalues["axis_object"]), 4),
            "axis_object_p": float(m3.pvalues["axis_object"]),
            "freq_cov_coef": round(float(m3.params["freq_cov"]), 4),
            "freq_cov_p": float(m3.pvalues["freq_cov"]),
            "n_clusters_lexeme": int(df3.lexeme.nunique()),
        }
    res3["formula"] = formula3
    res3["coding"] = ("axis_object: 1=객체(보충법),0=주체(-시-굴절). "
                      "freq_cov: 객체는 log_freq_ratio 중심화, 주체는 0(=객체평균빈도). "
                      "axis_object 계수 = 객체평균빈도에서 보충-굴절 격차(음수면 보충이 낮음).")
    res3["object_mean_log_freq_ratio"] = round(obj_mean_ratio, 4)
    out["task3_axis_after_freq"] = res3

    # 폴백 일관성 위해 cluster-robust OLS 도 항상 함께 보고
    m3_ols = fit_cluster_ols(df3, formula3, "lexeme")
    out["task3_axis_after_freq_clusterOLS"] = {
        "formula": formula3,
        "method": "cluster-robust OLS by lexeme (always-reported companion)",
        "axis_object_coef": round(float(m3_ols.params["axis_object"]), 4),
        "axis_object_se": round(float(m3_ols.bse["axis_object"]), 4),
        "axis_object_t": round(float(m3_ols.tvalues["axis_object"]), 4),
        "axis_object_p": float(m3_ols.pvalues["axis_object"]),
        "freq_cov_coef": round(float(m3_ols.params["freq_cov"]), 4),
        "freq_cov_p": float(m3_ols.pvalues["freq_cov"]),
        "r_squared": round(float(m3_ols.rsquared), 4),
        "n_clusters_lexeme": int(df3.lexeme.nunique()),
    }

    # ============================================================
    # 과제 3 robustness: 빈도대 매칭 부분집합 비교.
    #   주체 plain_freq(평형형 토큰빈도) 와 객체 pln_freq 를 같은 척도(log10)로.
    #   객체 보충쌍의 평형형 빈도(pln_freq)와 주체 어간빈도가 겹치는 대역에서
    #   axis 격차가 남는지. (두 빈도는 '평형형 절대빈도'로 동일 종류라 비교 가능;
    #    경어형 희귀성 자체와는 다른 통제임을 명시.)
    # ============================================================
    vf = json.load(open(ROOT / "data/freq/verb_freq.json"))
    lemma2plnfreq = {"주다": 7869, "말하다": 12245, "묻다": 349,
                     "보다": 10940, "만나다": 2013, "데리고 가다": 39}
    obj2 = obj.copy()
    obj2["plain_freq"] = obj2.verb_lemma.map(lemma2plnfreq)
    subj2 = subj.copy()
    # 주체 plain_freq 결측(0 또는 None) 제외하고 log10
    subj2 = subj2[subj2.plain_freq.notna() & (subj2.plain_freq.astype(float) > 0)].copy()
    subj2["plain_freq"] = subj2.plain_freq.astype(float)
    # 매칭 대역: 두 축 평형형빈도의 겹치는 범위
    lo = max(obj2.plain_freq.min(), subj2.plain_freq.min())
    hi = min(obj2.plain_freq.max(), subj2.plain_freq.max())
    obj_m = obj2[(obj2.plain_freq >= lo) & (obj2.plain_freq <= hi)]
    subj_m = subj2[(subj2.plain_freq >= lo) & (subj2.plain_freq <= hi)]
    matched = pd.concat([
        obj_m.assign(axis_object=1)[["final_logit_diff", "axis_object", "lexeme", "plain_freq"]],
        subj_m.assign(axis_object=0)[["final_logit_diff", "axis_object", "lexeme", "plain_freq"]],
    ], ignore_index=True)
    matched["log10_plain_freq"] = np.log10(matched.plain_freq.astype(float))
    mr = fit_cluster_ols(
        matched, "final_logit_diff ~ axis_object + log10_plain_freq", "lexeme")
    out["task3_robustness_freqmatched_subset"] = {
        "method": "cluster-robust OLS by lexeme on frequency-band-matched subset",
        "match_band_plain_freq": [float(lo), float(hi)],
        "n_object_matched": int(len(obj_m)),
        "n_subject_matched": int(len(subj_m)),
        "n_object_lexemes_matched": int(obj_m.lexeme.nunique()),
        "n_subject_lexemes_matched": int(subj_m.lexeme.nunique()),
        "axis_object_coef": round(float(mr.params["axis_object"]), 4),
        "axis_object_se": round(float(mr.bse["axis_object"]), 4),
        "axis_object_p": float(mr.pvalues["axis_object"]),
        "log10_plain_freq_coef": round(float(mr.params["log10_plain_freq"]), 4),
        "log10_plain_freq_p": float(mr.pvalues["log10_plain_freq"]),
        "note": ("평형형 절대빈도가 겹치는 대역에서도 axis_object 가 유의 음수면 "
                 "빈도통제 후 기제 해리 잔존. 단 이 통제는 '평형형 빈도'이며 "
                 "경어형 희귀성(log_freq_ratio)과는 다른 통제임."),
    }

    # ============================================================
    # 판정: '빈도+기제 혼합' 여부
    # ============================================================
    freq_explains = (out["task2_object_freq_regression"]["slope_p"] < 0.05)
    axis_survives_primary = (out["task3_axis_after_freq"]["axis_object_p"] < 0.05
                             and out["task3_axis_after_freq"]["axis_object_coef"] < 0)
    axis_survives_ols = (out["task3_axis_after_freq_clusterOLS"]["axis_object_p"] < 0.05
                         and out["task3_axis_after_freq_clusterOLS"]["axis_object_coef"] < 0)
    if axis_survives_primary and freq_explains:
        verdict = "FREQUENCY_PLUS_MECHANISM_MIXED"
        verdict_ko = ("빈도+기제 혼합: 경어형 희귀성(빈도)이 보충법 우세를 유의하게 "
                      "낮추지만(과제2), 빈도 통제 후에도 보충<굴절 축 격차가 잔존(과제3).")
    elif axis_survives_primary and not freq_explains:
        verdict = "MECHANISM_DOMINANT"
        verdict_ko = ("기제 우위: 빈도공변량은 유의치 않고, 빈도 통제 후 보충<굴절 "
                      "축 격차가 잔존. 해리는 주로 기제(검색/인코딩) 차이.")
    elif (not axis_survives_primary) and freq_explains:
        verdict = "FREQUENCY_DOMINANT"
        verdict_ko = ("빈도 우위: 빈도 통제 시 축 격차가 사라짐. 해리는 대체로 "
                      "경어형 어휘 저빈도로 설명됨.")
    else:
        verdict = "INCONCLUSIVE"
        verdict_ko = "빈도도 축도 통제 후 유의치 않음(검정력/표본 한계 가능)."
    out["verdict"] = verdict
    out["verdict_ko"] = verdict_ko
    out["verdict_inputs"] = {
        "freq_explains_object_failure(task2 slope p<.05)": bool(freq_explains),
        "axis_gap_survives_freq_primary(p<.05 & coef<0)": bool(axis_survives_primary),
        "axis_gap_survives_freq_clusterOLS(p<.05 & coef<0)": bool(axis_survives_ols),
    }
    out["caveats"] = (
        "객체 어휘 6개(군집6)로 robust/혼합 SE는 소표본 한계. 빈도비(log_freq_ratio)는 "
        "객체 보충쌍에만 정의되어 결합모형에선 주체=객체평균으로 두어 axis 더미가 "
        "'객체평균빈도에서의 축 격차'를 추정(생성적 -시-엔 보충빈도비 부재 때문). "
        "빈도대 매칭은 '평형형 절대빈도' 기준이라 '경어형 희귀성' 통제와는 다른 보조 통제."
    )

    (ROOT / "analysis/output/stats_m5.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
