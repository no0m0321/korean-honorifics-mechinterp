"""대조 최소쌍 자극 v2 — 적대 검증이 지적한 교란을 통제(workflow wq0pkdp4s).

v1(pairs.jsonl) 대비 변경:
  1) '님' 단서 무력화: 경어 referent를 '님' 없는 친족어 중심으로(소수만 -님 직함). neutral은 -님 0.
     → "'님' 포함" 단일 문자열 규칙이 라벨을 예측하지 못하게(v1은 정확도 1.000으로 분리됐음).
  2) 명사 직독 차단: 높임 논항과 표적 동사 사이에 부사어를 삽입 → 측정 위치(동사 직전)가 높임 명사
     자체가 아니라 부사어가 되도록. 경어 지위의 '전방 운반'을 강제(특히 accusative).
  3) 자기참조(주어=객체 동일 지시) 배제.
  4) 어휘당 프레임 확대(프로브 안정성).

주의: referent 지위는 본질적으로 어휘적이므로 프로빙만으로 '추상 요구 인코딩'을 표면 어휘에서
완전히 분리할 수는 없다(인과 3·4단계가 본 근거). v2는 (i) 단일 형태소 '님' 지름길과 (ii) 명사
직독 누수를 제거해 프로빙 증거를 강화한다. dative는 높임 명사가 상류에 있어 1차 M1 증거로 쓴다.

산출: data/stimuli/pairs_v2.jsonl
실행: .venv/bin/python -m src.make_stimuli_v2
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .honor_axes import detect_object_hon, detect_subject_hon

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# ── 경어 referent: 친족어 중심('님' 없음) + 소수 직함('님') ───────────────────
HON_KINSHIP = ["할아버지", "할머니", "아버지", "어머니", "큰아버지", "작은아버지",
               "외할아버지", "외할머니", "큰어머니", "외삼촌", "이모부", "고모부",
               "장인어른", "시아버지", "시어머니", "외숙모"]          # '님' 없음
HON_TITLE = ["사장님", "교수님", "부장님", "선생님", "원장님", "회장님"]  # '님' 있음(소수)
# 경어 referent에서 직함('님') 비율을 ~1/4로 제한 → '님'이 라벨을 예측 못하게
def _hon_pool(rng):
    return HON_KINSHIP * 3 + HON_TITLE  # 친족:직함 ≈ 48:6 (가중표집으로 '님' 약 11%)

# neutral referent: '님' 전무
PLN_REF = ["동생", "친구", "후배", "사촌", "동기", "막내", "철수", "민수", "지수",
           "현우", "영희", "짝꿍", "룸메이트", "조카", "선우", "다은"]
# 객체높임 문장의 평어 주어(−시− 누수 차단, 1인칭 포함)
PLN_AGENT = ["나는", "내가", "민수가", "철수가", "지수가", "현우가", "후배가", "영희가"]
# 논항과 동사 사이 부사어(명사 직독 차단 — 측정 위치를 부사어로)
ADVERBS = ["어제", "오늘", "방금", "아까", "먼저", "결국", "다시", "조용히",
           "천천히", "정중히", "직접", "급히"]


def _has_batchim(s: str) -> bool:
    last = s[-1]
    if 0xAC00 <= ord(last) <= 0xD7A3:
        return (ord(last) - 0xAC00) % 28 != 0
    return True


def subj_particle(name, honored):
    if honored:
        return f"{name}께서"
    return f"{name}{'이' if _has_batchim(name) else '가'}"


def dat_particle(name, honored):
    return f"{name}께" if honored else f"{name}에게"


def acc_particle(name):
    return f"{name}{'을' if _has_batchim(name) else '를'}"


def obj_marker(noun):
    return f"{noun}{'을' if _has_batchim(noun) else '를'}"


# 주체높임 동사(생성적 -시-, 과거형) — v1과 동일 큐레이션
SUBJ_VERBS = [
    {"lemma": "읽다", "hon": "읽으셨다", "pln": "읽었다", "kind": "trans",
     "themes": ["신문", "책", "편지", "기사", "보고서", "소설", "잡지", "메모"]},
    {"lemma": "받다", "hon": "받으셨다", "pln": "받았다", "kind": "trans",
     "themes": ["선물", "상", "전화", "편지", "월급", "꽃다발", "소포", "표창"]},
    {"lemma": "찾다", "hon": "찾으셨다", "pln": "찾았다", "kind": "trans",
     "themes": ["열쇠", "안경", "지갑", "자료", "서류", "책", "사진", "통장"]},
    {"lemma": "입다", "hon": "입으셨다", "pln": "입었다", "kind": "trans",
     "themes": ["외투", "한복", "정장", "코트", "양복", "스웨터", "조끼", "두루마기"]},
    {"lemma": "보내다", "hon": "보내셨다", "pln": "보냈다", "kind": "trans",
     "themes": ["편지", "선물", "자료", "초대장", "문자", "이메일", "엽서", "택배"]},
    {"lemma": "기다리다", "hon": "기다리셨다", "pln": "기다렸다", "kind": "trans",
     "themes": ["손님", "버스", "결과", "소식", "연락", "답장", "차례", "기차"]},
    {"lemma": "준비하다", "hon": "준비하셨다", "pln": "준비했다", "kind": "trans",
     "themes": ["저녁", "행사", "회의", "발표", "선물", "여행", "잔치", "강연"]},
    {"lemma": "가다", "hon": "가셨다", "pln": "갔다", "kind": "intrans",
     "advx": ["회사에", "시장에", "병원에", "교회에", "고향에", "법원에", "절에", "성당에"]},
    {"lemma": "오다", "hon": "오셨다", "pln": "왔다", "kind": "intrans",
     "advx": ["회의에", "집에", "행사에", "교실에", "사무실에", "모임에", "식장에", "병실에"]},
    {"lemma": "웃다", "hon": "웃으셨다", "pln": "웃었다", "kind": "intrans",
     "advx": ["크게", "환하게", "한참", "밝게", "살짝", "오래", "조용히", "소리없이"]},
    {"lemma": "앉다", "hon": "앉으셨다", "pln": "앉았다", "kind": "intrans",
     "advx": ["소파에", "의자에", "앞줄에", "창가에", "바닥에", "마루에", "방석에", "툇마루에"]},
    {"lemma": "떠나다", "hon": "떠나셨다", "pln": "떠났다", "kind": "intrans",
     "advx": ["고향을", "회사를", "서울을", "집을", "마을을", "병원을", "학교를", "직장을"]},
    {"lemma": "일하다", "hon": "일하셨다", "pln": "일했다", "kind": "intrans",
     "advx": ["늦게까지", "주말에", "혼자", "밤새", "현장에서", "공장에서", "들에서", "묵묵히"]},
    {"lemma": "도착하다", "hon": "도착하셨다", "pln": "도착했다", "kind": "intrans",
     "advx": ["역에", "공항에", "호텔에", "집에", "회장에", "현장에", "병원에", "터미널에"]},
    {"lemma": "출발하다", "hon": "출발하셨다", "pln": "출발했다", "kind": "intrans",
     "advx": ["새벽에", "혼자", "정시에", "기차로", "버스로", "차로", "함께", "곧장"]},
    {"lemma": "참석하다", "hon": "참석하셨다", "pln": "참석했다", "kind": "intrans",
     "advx": ["회의에", "행사에", "모임에", "장례식에", "결혼식에", "총회에", "예배에", "공청회에"]},
]

# 객체높임 보충법(폐쇄 부류, 과거형) — v1과 동일 6쌍
OBJ_VERBS = [
    {"lemma_pln": "주다", "lemma_hon": "드리다", "hon": "드렸다", "pln": "주었다",
     "case": "dat", "themes": ["서류", "선물", "책", "편지", "자료", "용돈", "꽃", "약"]},
    {"lemma_pln": "말하다", "lemma_hon": "말씀드리다", "hon": "말씀드렸다", "pln": "말했다",
     "case": "dat", "themes": ["사실", "결과", "계획", "소식", "이유", "생각", "사정", "의견"]},
    {"lemma_pln": "묻다", "lemma_hon": "여쭈다", "hon": "여쭈었다", "pln": "물었다",
     "case": "dat", "themes": ["길", "안부", "이유", "방법", "사정", "의견", "성함", "근황"]},
    {"lemma_pln": "보다", "lemma_hon": "뵙다", "hon": "뵈었다", "pln": "보았다", "case": "acc"},
    {"lemma_pln": "만나다", "lemma_hon": "뵙다", "hon": "뵈었다", "pln": "만났다", "case": "acc"},
    {"lemma_pln": "데리고 가다", "lemma_hon": "모시고 가다", "hon": "모시고 갔다",
     "pln": "데리고 갔다", "case": "acc"},
]


def _lexeme_id(s):
    return s.replace(" ", "_")


def build_subject(rng):
    items, fid = [], 0
    for v in SUBJ_VERBS:
        slots = v["themes"] if v["kind"] == "trans" else v["advx"]
        for slot in slots:                      # 동사당 8 프레임
            hp = _hon_pool(rng)
            s_hon = rng.choice(hp)
            s_pln = rng.choice(PLN_REF)
            adv = rng.choice(ADVERBS)
            mid = obj_marker(slot) if v["kind"] == "trans" else slot
            frame = f"subj_f{fid:03d}"
            for cond, name, honored in (("honorific", s_hon, True), ("neutral", s_pln, False)):
                # 논항 → 목적/부사 → 부사어 → 동사 (부사어가 동사 직전 = 측정위치)
                prompt = f"{subj_particle(name, honored)} {mid} {adv} "
                items.append({
                    "item_id": f"{frame}_{cond[:3]}", "frame_id": frame,
                    "axis": "subject", "morphology": "inflection", "suppletive": False,
                    "cond": cond, "expected": "honorific" if honored else "plain",
                    "prompt": prompt, "honorific_target": v["hon"], "plain_target": v["pln"],
                    "verb_lemma": v["lemma"], "lexeme_id": f"subj:{_lexeme_id(v['lemma'])}",
                    "subject": name, "subject_honored": int(honored),
                    "object": None, "object_honored": None, "object_case": None,
                    "honored_is_title": int(honored and name in HON_TITLE),
                    "adverb": adv,
                    "control_flags": {"si_leak_blocked": True, "apjon_excluded": True,
                                      "noun_upstream": True, "self_ref_excluded": True},
                })
            fid += 1
    return items


def build_object(rng):
    items, fid = [], 0
    for v in OBJ_VERBS:
        for k in range(8):                      # 동사당 8 프레임
            agent = rng.choice(PLN_AGENT)
            agent_base = agent.rstrip("가이은는을를께서에게 ")
            hp = _hon_pool(rng)
            # 자기참조 배제: 객체가 주어와 같은 지시면 재추첨
            o_hon = rng.choice(hp)
            o_pln = rng.choice([p for p in PLN_REF if p != agent_base])
            adv = rng.choice(ADVERBS)
            frame = f"obj_f{fid:03d}"
            for cond, oname, honored in (("honorific", o_hon, True), ("neutral", o_pln, False)):
                if v["case"] == "dat":
                    theme = v["themes"][k % len(v["themes"])]
                    # 주어 객체-께 목적물 부사어 동사 (높임명사 상류, 부사어가 동사직전)
                    prompt = f"{agent} {dat_particle(oname, honored)} {obj_marker(theme)} {adv} "
                    obj_case = "dative"
                else:
                    prompt = f"{agent} {acc_particle(oname)} {adv} "
                    obj_case = "accusative"
                items.append({
                    "item_id": f"{frame}_{cond[:3]}", "frame_id": frame,
                    "axis": "object", "morphology": "suppletion", "suppletive": True,
                    "cond": cond, "expected": "honorific" if honored else "plain",
                    "prompt": prompt, "honorific_target": v["hon"], "plain_target": v["pln"],
                    "verb_lemma": v["lemma_pln"], "lexeme_id": f"obj:{_lexeme_id(v['lemma_pln'])}",
                    "subject": agent, "subject_honored": 0,
                    "object": oname, "object_honored": int(honored), "object_case": obj_case,
                    "honored_is_title": int(honored and oname in HON_TITLE),
                    "adverb": adv,
                    "control_flags": {"si_leak_blocked": True, "apjon_excluded": True,
                                      "plain_agent": True, "noun_upstream": True,
                                      "self_ref_excluded": True},
                })
            fid += 1
    return items


def diagnostics(items):
    """교란 통제가 작동하는지 자가 진단."""
    import re
    # (1) '님' 단일규칙 균형정확도(낮아야 함)
    for axis in ("subject", "object"):
        hon = [it for it in items if it["axis"] == axis and it["cond"] == "honorific"]
        neu = [it for it in items if it["axis"] == axis and it["cond"] == "neutral"]
        def ref(it):
            return str(it["subject"] if axis == "subject" else it["object"])
        hon_nim = sum("님" in ref(it) for it in hon)
        neu_nim = sum("님" in ref(it) for it in neu)
        # 균형정확도: '님→hon, else→neu'
        bal = 0.5 * (hon_nim / max(len(hon), 1)) + 0.5 * (1 - neu_nim / max(len(neu), 1))
        print(f"  [{axis}] '님' 포함 hon {hon_nim}/{len(hon)} · neu {neu_nim}/{len(neu)} "
              f"→ '님'규칙 균형정확도 {bal:.3f} (1.0이면 완전교란; 0.5 목표)")
    # (2) 측정 위치(동사 직전 어절)가 높임 명사인가 — 아니어야(부사어여야)
    bad_pos = 0
    for it in items:
        last_word = it["prompt"].strip().split()[-1]
        ref = str(it["subject"] if it["axis"] == "subject" else it["object"])
        if ref and ref in last_word:
            bad_pos += 1
    print(f"  측정위치(동사직전)가 높임명사인 항목: {bad_pos} (0 목표 — 부사어가 동사직전)")
    # (3) 자기참조
    self_ref = sum(1 for it in items if it["axis"] == "object" and it["object"]
                   and str(it["object"]) == it["subject"].rstrip("가이은는을를께서에게 "))
    print(f"  자기참조(주어=객체): {self_ref} (0 목표)")


def main():
    rng = random.Random(SEED)
    subj, obj = build_subject(rng), build_object(rng)
    alli = subj + obj
    out = ROOT / "data/stimuli/pairs_v2.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in alli) + "\n",
                   encoding="utf-8")
    print(f"주체높임: {len(subj)}자극 = {len({x['frame_id'] for x in subj})}쌍, "
          f"어휘 {len({x['lexeme_id'] for x in subj})}")
    print(f"객체높임: {len(obj)}자극 = {len({x['frame_id'] for x in obj})}쌍, "
          f"어휘 {len({x['lexeme_id'] for x in obj})}")
    print(f"총 {len(alli)}자극 → {out.relative_to(ROOT)}\n예시:")
    for x in (subj[0], obj[0], obj[48]):
        print(f"  [{x['axis']}/{x['cond']:9}] {x['prompt']}「{x['honorific_target']}|{x['plain_target']}」")
    print("\n교란 통제 진단:")
    diagnostics(alli)
    # 탐지기 자가검증
    errs = 0
    for it in alli:
        det = detect_subject_hon if it["axis"] == "subject" else detect_object_hon
        if det(it["prompt"] + it["honorific_target"]) != 1 or det(it["prompt"] + it["plain_target"]) != 0:
            errs += 1
    print(f"\n탐지기 자가검증 불일치: {errs}건")


if __name__ == "__main__":
    main()
