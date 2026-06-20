"""1단계 선형 프로빙 (가설 M1·M6).

각 층 잔차흐름이 '경어 촉발 = 고위 논항의 존재'를 선형 디코딩 가능한지 측정한다.
핵심 검증: 만약 보충법(객체) 문맥에서도 프로브가 성공하면 '요구는 인코딩됨' → 실패의
책임은 인코딩이 아니라 후기 검색에 있다는 1차 증거(계획서 §6.1).

과대표현 차단(Hewitt & Liang 2019):
  - 어휘 항목(lexeme_id) 단위로 train/test 분할(GroupKFold) → 프로브가 단어를 못 외움
  - 통제 과제: 어휘별 무작위 고정 레이블. 어휘 분할에서는 일반화 불가 → 통제 정확도≈우연
  - 선택도 = 과제 정확도 − 통제 정확도 (실제 인코딩의 순수 측정)
  - 부트스트랩 95% CI가 0을 포함하지 않으면 M1 채택(선택도 > 0.10 동반)

축별(subject=굴절 / object=보충법) 따로 추정하여 RQ1을 직접 답한다.

입력: cache_acts.py가 저장한 acts/<model>.npz
실행: .venv/bin/python -m src.probe <model_key>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 0
N_BOOT = 1000


def _fit_layer(X, y, groups, y_ctrl, n_splits=5):
    """한 층에서 어휘분할 교차검증 정확도(과제·통제) 반환."""
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
        c2 = LogisticRegression(max_iter=2000, C=0.5)
        c2.fit(Xtr, y_ctrl[tr])
        ctrl_correct[te] = c2.predict(Xte) == y_ctrl[te]
    return task_correct, ctrl_correct


def _bootstrap_ci(task_correct, ctrl_correct, groups, n_boot=N_BOOT):
    """어휘(그룹) 단위 부트스트랩으로 선택도 95% CI."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    sels = []
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        sels.append(task_correct[idx].mean() - ctrl_correct[idx].mean())
    sels = np.array(sels)
    return float(np.percentile(sels, 2.5)), float(np.percentile(sels, 97.5))


def probe_axis(X_all, y, groups, axis_mask):
    """한 축의 층별 프로빙. X_all: [n_items, n_layers, hidden]."""
    rng = np.random.default_rng(SEED)
    X = X_all[axis_mask]
    y_a = y[axis_mask]
    g_a = groups[axis_mask]
    # 통제 레이블: 어휘별 무작위 0/1 고정
    uniq = np.unique(g_a)
    ctrl_map = {g: int(rng.integers(0, 2)) for g in uniq}
    y_ctrl = np.array([ctrl_map[g] for g in g_a])

    n_layers = X.shape[1]
    rows = []
    for L in range(n_layers):
        tc, cc = _fit_layer(X[:, L, :], y_a, g_a, y_ctrl)
        task_acc, ctrl_acc = tc.mean(), cc.mean()
        sel = task_acc - ctrl_acc
        lo, hi = _bootstrap_ci(tc, cc, g_a)
        rows.append({"layer": L, "task_acc": float(task_acc), "ctrl_acc": float(ctrl_acc),
                     "selectivity": float(sel), "ci_lo": lo, "ci_hi": hi,
                     "m1_pass": bool(sel > 0.10 and lo > 0)})
    return rows


def main():
    import json

    import os
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    suf = f"_{os.environ.get('STIM', '')}" if os.environ.get("STIM") else ""
    npz = np.load(ROOT / f"analysis/output/acts_{key}{suf}.npz", allow_pickle=True)
    X = npz["X"]                       # [n_items, n_layers, hidden]
    y = npz["y"]                       # cond: honorific=1 / neutral=0
    groups = npz["lexeme"]             # 어휘 id (문자열)
    axis = npz["axis"]                 # 'subject' / 'object'

    out = {"model": key, "n_layers": int(X.shape[1])}
    for ax in ("subject", "object"):
        mask = axis == ax
        rows = probe_axis(X, y, groups, mask)
        peak = max(rows, key=lambda r: r["selectivity"])
        out[ax] = rows
        print(f"[{key}/{ax}] 정점 층 L{peak['layer']} 선택도={peak['selectivity']:.3f} "
              f"CI[{peak['ci_lo']:.3f},{peak['ci_hi']:.3f}] "
              f"M1통과층={sum(r['m1_pass'] for r in rows)}/{len(rows)}")

    outp = ROOT / f"analysis/output/probe_{key}{suf}.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
