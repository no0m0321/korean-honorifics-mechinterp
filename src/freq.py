"""M5 빈도 집계: 보충법 동사쌍의 평형–경어 로그빈도 비.

'보충법 실패가 단순 희소 어휘 때문인가, 빈도 통제 후에도 남는 구조적 기제인가'(M5)를
분리하기 위한 공변량. 공개 한국어 뉴스 코퍼스(naver-news)에서 Kiwi 형태소 분석으로 동사
어간 빈도를 집계한다. 평형 어휘(주다)는 빈번, 보충법 어휘(드리다)는 희소하리라 예측.

산출: data/freq/verb_freq.json
실행: .venv/bin/python -m src.freq
"""
from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

# 보충법 동사쌍 (평형 어간, 경어 어간) — make_stimuli/honor_axes와 일치
OBJ_PAIRS = [("주", "드리"), ("말하", "말씀드리"), ("묻", "여쭈"),
             ("보", "뵙"), ("만나", "뵙"), ("데리", "모시")]
# 주체높임 생성적 동사 어간 (통제: -시- 굴절은 어휘 무관 생성적)
SUBJ_STEMS = ["읽", "받", "찾", "입", "보내", "기다리", "준비하", "가", "오", "웃",
              "앉", "떠나", "일하", "도착하", "출발하", "참석하"]

# 표면 텍스트 보조 매칭(복합·이형태·Kiwi 분리 누락 보완).
# Kiwi가 어간으로 복원 못하는 항목: '말하다'=말(NNG)+하(XSV), '데리고가다'=데려가(VV) 등.
SURFACE = {
    "주": None, "드리": ["드리"],
    "말하": ["말하", "말했", "말한", "말할"], "말씀드리": ["말씀드리", "말씀 드리"],
    "묻": None, "여쭈": ["여쭈", "여쭙", "여쭤"],
    "보": None, "뵙": ["뵙", "뵈"],
    "만나": None, "데리": ["데려가", "데리고", "데려"], "모시": ["모시"],
}


def count_corpus(limit: int = 22194):
    from datasets import load_dataset
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    ds = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
    stem_cnt: Counter = Counter()
    surf_text_cnt: Counter = Counter()
    n_verb = 0
    n = min(limit, len(ds))
    surf_targets = {k: v for k, v in SURFACE.items() if v}
    for i in range(n):
        text = ds[i].get("document") or ""
        for t in kiwi.tokenize(text):
            if t.tag[:2] in ("VV", "VX"):  # VV-I/VV-R 등 불규칙 태그 포함
                stem_cnt[t.form] += 1
                n_verb += 1
        # 표면 보조 카운트(Kiwi 어간 복원 누락 항목)
        for stem, surfs in surf_targets.items():
            surf_text_cnt[stem] += sum(text.count(s) for s in surfs)
    return stem_cnt, surf_text_cnt, n_verb, n


def main():
    stem_cnt, surf_cnt, n_verb, n_doc = count_corpus()

    def freq(stem: str) -> int:
        base = stem_cnt.get(stem, 0)
        # Kiwi 어간 복원이 안 되는 항목은 표면 카운트로 보정(더 큰 값 채택)
        if stem in surf_cnt and surf_cnt[stem] > 0:
            base = max(base, surf_cnt[stem])
        return base

    pairs = []
    for pln, hon in OBJ_PAIRS:
        fp, fh = freq(pln), freq(hon)
        pairs.append({
            "pair": f"{pln}→{hon}",
            "pln_stem": pln, "hon_stem": hon,
            "pln_freq": fp, "hon_freq": fh,
            "log_freq_ratio": round(math.log((fp + 1) / (fh + 1)), 3),  # 양수=평형이 더 빈번
        })
    subj = {s: freq(s) for s in SUBJ_STEMS}

    out = {
        "corpus": "daekeun-ml/naver-news-summarization-ko",
        "n_doc": n_doc, "n_verb_token": n_verb,
        "object_suppletive_pairs": pairs,
        "subject_stems": subj,
        "note": "log_freq_ratio>0 이면 평형 어휘가 보충법 어휘보다 빈번(=보충법 희소).",
    }
    outp = ROOT / "data/freq/verb_freq.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"코퍼스 {n_doc}문서, 동사 토큰 {n_verb:,}")
    print("보충법 동사쌍 로그빈도비(양수=보충법 희소):")
    for p in pairs:
        print(f"  {p['pair']:14} 평형 {p['pln_freq']:>6} / 경어 {p['hon_freq']:>5}  "
              f"logratio={p['log_freq_ratio']:+.2f}")
    print("저장:", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
