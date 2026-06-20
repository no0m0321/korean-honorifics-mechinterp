"""자극 v3 — 주체 보충법 추가로 'suppletion vs object' 교란 분리 (핵심 설계 개선).

v2는 굴절=주체·보충법=객체로 두 요인(형태론 × 통사역할)이 완전 공선이다. v3는 **주체 높임
보충법**(자다→주무시다, 먹다→드시다, 있다→계시다 등)을 추가해 분리한다:
  - subject (굴절): 읽으셨다 / 읽었다          [주체 · 굴절]
  - subject_suppl (보충법): 주무셨다 / 잤다     [주체 · 보충법]  ← 신규
  - object (보충법): 드렸다 / 주었다            [객체 · 보충법]
핵심 검정: 주체 보충법도 객체 보충법처럼 실패하면 → 실패는 *보충법(어휘 인출)* 때문(통사역할
무관). 주체 보충법이 굴절처럼 성공하면 → 실패는 *객체* 때문. 논문의 'inflection vs suppletion'
프레임의 결정적 시험.

v2의 인벤토리·교란 통제(친족어 중심, 부사어 삽입, 자기참조 배제)를 그대로 계승.
산출: data/stimuli/pairs_v3.jsonl
실행: .venv/bin/python -m src.make_stimuli_v3
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .honor_axes import detect_object_hon, detect_subject_hon
from .make_stimuli_v2 import (ADVERBS, _hon_pool, PLN_REF, build_object,
                              build_subject, obj_marker, subj_particle)

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# 주체 높임 보충법(경어형이 별도 어휘) — 과거형 큐레이션
SUBJ_SUPPL = [
    {"lemma": "자다", "hon": "주무셨다", "pln": "잤다", "kind": "intrans",
     "advx": ["일찍", "곤히", "오래", "푹", "늦게까지", "편히", "낮에", "깊이"]},
    {"lemma": "먹다", "hon": "드셨다", "pln": "먹었다", "kind": "trans",
     "themes": ["저녁", "약", "과일", "점심", "국", "밥", "간식", "떡"]},
    {"lemma": "마시다", "hon": "드셨다", "pln": "마셨다", "kind": "trans",
     "themes": ["차", "물", "커피", "약", "주스", "녹차", "국물", "음료"]},
    {"lemma": "있다", "hon": "계셨다", "pln": "있었다", "kind": "intrans",
     "advx": ["집에", "방에", "거실에", "서재에", "마당에", "자리에", "안방에", "사랑채에"]},
    {"lemma": "아프다", "hon": "편찮으셨다", "pln": "아팠다", "kind": "intrans",
     "advx": ["많이", "요즘", "며칠째", "오래", "심하게", "한동안", "계속", "밤새"]},
]


def build_subject_suppl(rng):
    items, fid = [], 0
    for v in SUBJ_SUPPL:
        slots = v["themes"] if v["kind"] == "trans" else v["advx"]
        for slot in slots:
            s_hon = rng.choice(_hon_pool(rng))
            s_pln = rng.choice(PLN_REF)
            adv = rng.choice(ADVERBS)
            mid = obj_marker(slot) if v["kind"] == "trans" else slot
            frame = f"ssup_f{fid:03d}"
            for cond, name, honored in (("honorific", s_hon, True), ("neutral", s_pln, False)):
                prompt = f"{subj_particle(name, honored)} {mid} {adv} "
                items.append({
                    "item_id": f"{frame}_{cond[:3]}", "frame_id": frame,
                    "axis": "subject_suppl", "morphology": "suppletion", "suppletive": True,
                    "cond": cond, "expected": "honorific" if honored else "plain",
                    "prompt": prompt, "honorific_target": v["hon"], "plain_target": v["pln"],
                    "verb_lemma": v["lemma"], "lexeme_id": f"ssup:{v['lemma']}",
                    "subject": name, "subject_honored": int(honored),
                    "object": None, "object_honored": None, "object_case": None,
                    "honored_is_title": 0, "adverb": adv,
                    "control_flags": {"si_leak_blocked": True, "apjon_excluded": True,
                                      "noun_upstream": True, "self_ref_excluded": True},
                })
            fid += 1
    return items


def main():
    rng = random.Random(SEED)
    subj = build_subject(rng)          # 주체 굴절 (v2)
    ssup = build_subject_suppl(rng)    # 주체 보충법 (신규)
    obj = build_object(rng)            # 객체 보충법 (v2)
    alli = subj + ssup + obj
    out = ROOT / "data/stimuli/pairs_v3.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in alli) + "\n",
                   encoding="utf-8")
    for name, grp in [("subject(굴절)", subj), ("subject_suppl(주체보충법)", ssup),
                      ("object(객체보충법)", obj)]:
        print(f"{name}: {len(grp)}자극 = {len({x['frame_id'] for x in grp})}쌍, "
              f"어휘 {len({x['lexeme_id'] for x in grp})}")
    print(f"총 {len(alli)} → {out.relative_to(ROOT)}\n예시(주체보충법):")
    for x in ssup[:2]:
        print(f"  [{x['cond']:9}] {x['prompt']}「{x['honorific_target']}|{x['plain_target']}」")
    # 탐지기 자가검증
    errs = 0
    for it in alli:
        det = detect_object_hon if it["axis"] == "object" else detect_subject_hon
        if det(it["prompt"] + it["honorific_target"]) != 1 or det(it["prompt"] + it["plain_target"]) != 0:
            errs += 1
    print(f"탐지기 자가검증 불일치: {errs}건")


if __name__ == "__main__":
    main()
