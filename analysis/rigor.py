"""교정 분석 정본(적대 검증 반영). 버전 인식(STIM=v2 → _v2 파일).

적대 검증(workflow wq0pkdp4s)이 확정한 결함을 교정해 재현 가능하게 한다:
  M1: 객체 선택도를 pooled/dative/accusative로 분리(dative=높임명사 상류, 1차 증거). 통제과제는
      소어휘 퇴화를 막기 위해 다수 시드 평균. 표면 문자 n-gram 베이스라인을 동반 보고(프로빙이
      표면을 넘어서는지 가늠 — 단, 지위는 본질적으로 어휘적이라 인과단계가 본 근거).
  M2: 주체 vs 객체 최종 logit_diff를 '어휘 단위'로 정직 검정(Welch·순열·Cohen d). few-cluster
      과대정밀(OLS p=5e-14) 대신 어휘평균 단위.
  M5: 객체 내부에서 최종 logit_diff ~ log_freq_ratio(비순환). 어휘별 표로 빈도 의존성 공개.

실행: STIM=v2 .venv/bin/python analysis/rigor.py <model_key>
산출: analysis/output/rigor_<key>[_v2].json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/output"
KEY = sys.argv[1] if len(sys.argv) > 1 else "exaone"
STIM = os.environ.get("STIM", "")
SUF = f"_{STIM}" if STIM else ""
PAIRS = "data/stimuli/pairs_v2.jsonl" if STIM == "v2" else "data/stimuli/pairs.jsonl"

LEMMA2STEM = {"주다": "주", "말하다": "말하", "묻다": "묻", "보다": "보",
              "만나다": "만나", "데리고 가다": "데리"}


def _task_acc(XL, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    ok = np.zeros(len(y), bool)
    for tr, te in gkf.split(XL, y, groups):
        sc = StandardScaler().fit(XL[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(XL[tr]), y[tr])
        ok[te] = clf.predict(sc.transform(XL[te])) == y[te]
    return ok.mean()


def _ctrl_acc(XL, groups, seeds):
    """어휘별 무작위 레이블 통제 정확도, 다수 시드 평균(소어휘 퇴화 완화)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    uniq = np.unique(groups)
    accs = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        cmap = {g: int(rng.integers(0, 2)) for g in uniq}
        yc = np.array([cmap[g] for g in groups])
        cc = np.zeros(len(groups), bool)
        for tr, te in gkf.split(XL, yc, groups):
            if len(np.unique(yc[tr])) < 2:
                cc[te] = yc[te] == yc[tr][0]
                continue
            sc = StandardScaler().fit(XL[tr])
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(XL[tr]), yc[tr])
            cc[te] = clf.predict(sc.transform(XL[te])) == yc[te]
        accs.append(cc.mean())
    return float(np.mean(accs)), float(np.std(accs))


def m1_selectivity(X, y, groups, n_layers, refnoun=None):
    """2-pass: 1차 task_acc로 정점 층, 2차 정점에서 다수 시드 통제.
    refnoun 제공 시 '명사 분리'(미관측 referent로 일반화) task_acc도 정점 층에서 보고 —
    어휘(동사) 분할이 못 막는 명사 암기를 차단하는 결정적 추상표상 검정."""
    tas = [(_task_acc(X[:, L, :], y, groups), L) for L in range(n_layers)]
    ta, L = max(tas)
    ca, cs = _ctrl_acc(X[:, L, :], groups, seeds=25)
    out = {"layer": int(L), "task_acc": round(float(ta), 3), "ctrl_acc": round(ca, 3),
           "ctrl_sd": round(cs, 3), "selectivity": round(float(ta) - ca, 3)}
    if refnoun is not None and len(np.unique(refnoun)) >= 4:
        # 명사 분리 일반화: train/test referent 인물이 겹치지 않음
        nd_task = _task_acc(X[:, L, :], y, refnoun)
        nd_ctrl, _ = _ctrl_acc(X[:, L, :], refnoun, seeds=25)
        out["noun_disjoint_task_acc"] = round(float(nd_task), 3)
        out["noun_disjoint_selectivity"] = round(float(nd_task) - nd_ctrl, 3)
        out["n_referent"] = int(len(np.unique(refnoun)))
    return out


def surface_baseline(prompts, y, groups):
    """프롬프트 문자 n-gram만으로 라벨 예측(표면 베이스라인, 어휘분할)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    nsp = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=nsp)
    correct = np.zeros(len(y), bool)
    for tr, te in gkf.split(prompts, y, groups):
        vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        Xtr = vec.fit_transform([prompts[i] for i in tr])
        Xte = vec.transform([prompts[i] for i in te])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
        correct[te] = clf.predict(Xte) == y[te]
    return round(float(correct.mean()), 3)


def main():
    acts = np.load(OUT / f"acts_{KEY}{SUF}.npz", allow_pickle=True)
    X, y, lex, axis = acts["X"], acts["y"], acts["lexeme"], acts["axis"]
    ocase = acts["object_case"]
    nL = X.shape[1]
    items = [json.loads(l) for l in (ROOT / PAIRS).read_text().splitlines()]
    prompts = np.array([it["prompt"] for it in items])
    refnoun = np.array([str(it["subject"] if it["axis"] == "subject" else it["object"])
                        for it in items])
    res = {"model": KEY, "stim": STIM or "v1", "n_layers": int(nL)}

    # ── M1: pooled / dative / accusative + 표면 베이스라인 ──
    m1 = {}
    for name, mask in [("subject", axis == "subject"),
                       ("object_pooled", axis == "object"),
                       ("object_dative", (axis == "object") & (ocase == "dative")),
                       ("object_accusative", (axis == "object") & (ocase == "accusative"))]:
        if mask.sum() == 0:
            continue
        peak = m1_selectivity(X[mask], y[mask], lex[mask], nL, refnoun=refnoun[mask])
        peak["surface_ngram_acc"] = surface_baseline(prompts[mask], y[mask], lex[mask])
        peak["n"] = int(mask.sum())
        peak["n_lexeme"] = int(len(np.unique(lex[mask])))
        m1[name] = peak
    res["M1"] = m1

    # ── M2: 어휘 단위 정직 통계 (최종 logit_diff, honorific) ──
    lens = np.load(OUT / f"lens_{KEY}{SUF}.npz", allow_pickle=True)
    diff, laxis, lcond, llex = lens["diff"], lens["axis"], lens["cond"], lens["lexeme"]
    finalLD = diff[:, -1]
    honm = (lcond == "honorific") & np.isfinite(finalLD)

    def lex_means(ax):
        m = honm & (laxis == ax)
        out = {}
        for lx in np.unique(llex[m]):
            out[lx] = float(np.nanmean(finalLD[m & (llex == lx)]))
        return out

    sub_means = lex_means("subject")
    obj_means = lex_means("object")
    sv, ov = np.array(list(sub_means.values())), np.array(list(obj_means.values()))
    welch = stats.ttest_ind(sv, ov, equal_var=False)
    # 순열검정(어휘평균 라벨 셔플)
    allv = np.concatenate([sv, ov])
    labels = np.array([1] * len(sv) + [0] * len(ov))
    obsd = sv.mean() - ov.mean()
    rng = np.random.default_rng(0)
    perm = sum(abs((allv[(p := rng.permutation(labels)) == 1]).mean()
                   - allv[p == 0].mean()) >= abs(obsd) for _ in range(20000)) / 20000
    pooled_sd = np.sqrt(((len(sv) - 1) * sv.std(ddof=1) ** 2 + (len(ov) - 1) * ov.std(ddof=1) ** 2)
                        / (len(sv) + len(ov) - 2))
    res["M2"] = {
        "subject_final_LD_mean": round(float(np.nanmean(finalLD[honm & (laxis == 'subject')])), 3),
        "object_final_LD_mean": round(float(np.nanmean(finalLD[honm & (laxis == 'object')])), 3),
        "subject_cross_frac": round(float((finalLD[honm & (laxis == 'subject')] > 0).mean()), 3),
        "object_cross_frac": round(float((finalLD[honm & (laxis == 'object')] > 0).mean()), 3),
        "lexeme_welch_t": round(float(welch.statistic), 3),
        "lexeme_welch_p": float(welch.pvalue),
        "lexeme_permutation_p": float(perm),
        "cohen_d_lexeme": round(float(obsd / pooled_sd), 3),
        "subject_lexeme_means": {k: round(v, 2) for k, v in sub_means.items()},
        "object_lexeme_means": {k: round(v, 2) for k, v in obj_means.items()},
    }

    # ── M5: 객체 내부 빈도 의존성(비순환) ──
    freq = json.loads((ROOT / "data/freq/verb_freq.json").read_text())
    stem2lfr = {p["pln_stem"]: p["log_freq_ratio"] for p in freq["object_suppletive_pairs"]}
    lemma_of = {it["lexeme_id"]: it["verb_lemma"] for it in items if it["axis"] == "object"}
    rows = []
    for lx, ld in obj_means.items():
        lemma = lemma_of.get(lx, "")
        lfr = stem2lfr.get(LEMMA2STEM.get(lemma, ""), None)
        rows.append({"lexeme": lx, "final_LD": round(ld, 2), "log_freq_ratio": lfr})
    valid = [(r["log_freq_ratio"], r["final_LD"]) for r in rows if r["log_freq_ratio"] is not None]
    if len(valid) >= 3:
        xf, yf = zip(*valid)
        r, p = stats.pearsonr(xf, yf)
        res["M5"] = {"object_lexeme_freq_vs_LD": rows,
                     "pearson_r": round(float(r), 3), "pearson_p": float(p),
                     "note": "음의 r = 희소(고 log_freq_ratio)할수록 logit_diff 낮음(실패). 6어휘로 빈도-독립 기제 식별 불가."}
    else:
        res["M5"] = {"object_lexeme_freq_vs_LD": rows, "note": "유효 어휘 부족"}

    outp = OUT / f"rigor_{KEY}{SUF}.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
