"""대조 최소쌍 자극 생성기 (화이트박스 기계적 해석용).

연구계획서 §4 (자극 설계) 구현. 선행 김승우(2026a)의 블랙박스 행동측정과 달리,
여기서는 **표적 동사 위치 직전까지를 프롬프트로 주고** 경어형/평형형 표면형의
로짓·활성값을 비교한다(프로빙·로짓렌즈·패칭·조향의 공통 입력).

대조 최소쌍 원리(§4.1): 경어를 촉발하는 조건과 촉발하지 않는 조건이 단 하나의
요인(고위 논항 대 또래 논항)에서만 달라야 한다. 명제 내용·어순·길이·격틀은 동일.

두 축(§4.2):
  - 주체높임 -시- (굴절·생성적): 어간 공유, 어미에서 분기 (읽으셨다 / 읽었다).
    생성적이므로 규칙 동사로 항목 확장.
  - 객체높임 보충법 (어휘·비규칙): 어간 자체가 분기 (드렸다 / 주었다).
    한국어에서 사실상 폐쇄 부류 — 이 소수성 자체가 본 연구의 핵심.

교란 통제(§4.2):
  - 객체높임 문장의 주어는 평어 인물 고정 → 주체 -시- 누수 차단.
  - 압존법·간접 높임·무생물 주어·2인칭 주어는 사전 배제(인벤토리 자체로).

활용형은 자동 생성하지 않고 동사별 완전 표면형을 큐레이션한다(make_honor_data.py의
memo 원칙 계승: Kiwi 자동 활용 대신 정확한 표면형 명시). 시제는 과거 평서 -았/었다체로
통일(계획서 표2의 '드렸다/주었다'와 일관, 보충법 자연도 최상).

산출: data/stimuli/pairs.jsonl  (한 줄 = 한 자극)
검증: 각 표면형이 honor_axes 탐지기와 일치하는지 자가검사.
실행: .venv/bin/python -m src.make_stimuli
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .honor_axes import detect_object_hon, detect_subject_hon

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# ── 논항 인벤토리 ────────────────────────────────────────────────────────────
# 모두 3인칭 존대가능 인물 / 또래 인물. 무생물·2인칭 배제(인벤토리로 통제).
HON_SUBJ = ["할아버지", "할머니", "교수님", "사장님", "선생님", "부장님",
            "아버지", "어머니", "원장님", "회장님", "이사님", "장관님"]
PLN_SUBJ = ["동생", "친구", "민수", "철수", "후배", "누나",
            "조카", "사촌", "지수", "현우", "동기", "막내"]
# 객체높임 문장의 주어 — 평어 인물 고정(-시- 누수 차단). 1인칭 포함.
PLN_AGENT = ["나는", "내가", "민수가", "철수가", "지수가", "현우가", "후배가", "동생이"]
# 객체(수혜자/대상)
HON_OBJ = ["할아버지", "할머니", "교수님", "사장님", "선생님", "부장님",
           "원장님", "회장님", "큰아버지", "이사님", "고객님", "손님"]
PLN_OBJ = ["동생", "친구", "민수", "철수", "후배", "조카", "지수", "현우", "동기", "막내"]


# ── 받침 판정(조사 선택) ──────────────────────────────────────────────────────
def _has_batchim(s: str) -> bool:
    last = s[-1]
    if 0xAC00 <= ord(last) <= 0xD7A3:
        return (ord(last) - 0xAC00) % 28 != 0
    return True  # 비한글은 받침 있다고 보수적 처리


def subj_particle(name: str, honored: bool) -> str:
    if honored:
        return f"{name}께서"
    return f"{name}{'이' if _has_batchim(name) else '가'}"


def dat_particle(name: str, honored: bool) -> str:  # 여격 -께/-에게
    return f"{name}께" if honored else f"{name}에게"


def acc_particle(name: str) -> str:  # 대격 -을/-를 (높임 무관 — 보충법만이 단서)
    return f"{name}{'을' if _has_batchim(name) else '를'}"


def obj_marker(noun: str) -> str:  # 목적물 -을/-를
    return f"{noun}{'을' if _has_batchim(noun) else '를'}"


# ── 주체높임 동사 (생성적 -시-, 과거형 완전 큐레이션) ──────────────────────────
# kind: intrans(자동사, 부사어) | trans(타동사, 목적어). hon=경어형, pln=평형형.
SUBJ_VERBS = [
    {"lemma": "읽다", "hon": "읽으셨다", "pln": "읽었다", "kind": "trans",
     "themes": ["신문", "책", "편지", "기사", "보고서", "소설"]},
    {"lemma": "받다", "hon": "받으셨다", "pln": "받았다", "kind": "trans",
     "themes": ["선물", "상", "전화", "편지", "월급", "꽃다발"]},
    {"lemma": "찾다", "hon": "찾으셨다", "pln": "찾았다", "kind": "trans",
     "themes": ["열쇠", "안경", "지갑", "자료", "서류", "책"]},
    {"lemma": "입다", "hon": "입으셨다", "pln": "입었다", "kind": "trans",
     "themes": ["외투", "한복", "정장", "코트", "양복", "스웨터"]},
    {"lemma": "보내다", "hon": "보내셨다", "pln": "보냈다", "kind": "trans",
     "themes": ["편지", "선물", "자료", "초대장", "문자", "이메일"]},
    {"lemma": "기다리다", "hon": "기다리셨다", "pln": "기다렸다", "kind": "trans",
     "themes": ["손님", "버스", "결과", "소식", "연락", "답장"]},
    {"lemma": "준비하다", "hon": "준비하셨다", "pln": "준비했다", "kind": "trans",
     "themes": ["저녁", "행사", "회의", "발표", "선물", "여행"]},
    {"lemma": "가다", "hon": "가셨다", "pln": "갔다", "kind": "intrans",
     "advs": ["회사에", "시장에", "병원에", "교회에", "고향에", "산책을"]},
    {"lemma": "오다", "hon": "오셨다", "pln": "왔다", "kind": "intrans",
     "advs": ["일찍", "방금", "회의에", "집에", "행사에", "늦게"]},
    {"lemma": "웃다", "hon": "웃으셨다", "pln": "웃었다", "kind": "intrans",
     "advs": ["크게", "환하게", "조용히", "한참", "밝게", "살짝"]},
    {"lemma": "앉다", "hon": "앉으셨다", "pln": "앉았다", "kind": "intrans",
     "advs": ["소파에", "의자에", "앞줄에", "창가에", "조용히", "바닥에"]},
    {"lemma": "떠나다", "hon": "떠나셨다", "pln": "떠났다", "kind": "intrans",
     "advs": ["일찍", "어제", "급히", "혼자", "새벽에", "조용히"]},
    {"lemma": "일하다", "hon": "일하셨다", "pln": "일했다", "kind": "intrans",
     "advs": ["늦게까지", "주말에", "열심히", "혼자", "밤새", "조용히"]},
    {"lemma": "도착하다", "hon": "도착하셨다", "pln": "도착했다", "kind": "intrans",
     "advs": ["일찍", "정시에", "방금", "늦게", "무사히", "먼저"]},
    {"lemma": "출발하다", "hon": "출발하셨다", "pln": "출발했다", "kind": "intrans",
     "advs": ["새벽에", "먼저", "급히", "일찍", "정시에", "혼자"]},
    {"lemma": "참석하다", "hon": "참석하셨다", "pln": "참석했다", "kind": "intrans",
     "advs": ["회의에", "행사에", "모임에", "장례식에", "결혼식에", "직접"]},
]

# ── 객체높임 보충법 (폐쇄 부류, 과거형 완전 큐레이션) ──────────────────────────
# case: dat(여격 -께/-에게 + 목적물)  |  acc(대격 -을/-를, 격조사 단서 없음)
OBJ_VERBS = [
    {"lemma_pln": "주다", "lemma_hon": "드리다", "hon": "드렸다", "pln": "주었다",
     "case": "dat", "themes": ["서류", "선물", "책", "편지", "자료", "용돈", "꽃"]},
    {"lemma_pln": "말하다", "lemma_hon": "말씀드리다", "hon": "말씀드렸다", "pln": "말했다",
     "case": "dat", "themes": ["사실", "결과", "계획", "소식", "이유", "생각"]},
    {"lemma_pln": "묻다", "lemma_hon": "여쭈다", "hon": "여쭈었다", "pln": "물었다",
     "case": "dat", "themes": ["길", "안부", "이유", "방법", "사정", "의견"]},
    {"lemma_pln": "보다", "lemma_hon": "뵙다", "hon": "뵈었다", "pln": "보았다",
     "case": "acc"},
    {"lemma_pln": "만나다", "lemma_hon": "뵙다", "hon": "뵈었다", "pln": "만났다",
     "case": "acc"},
    {"lemma_pln": "데리고 가다", "lemma_hon": "모시고 가다", "hon": "모시고 갔다",
     "pln": "데리고 갔다", "case": "acc"},
]


def _lexeme_id(s: str) -> str:
    return s.replace(" ", "_")


def build_subject(rng: random.Random) -> list[dict]:
    """주체높임 -시- 최소쌍: [주어-께서/-가] [목적어/부사어] [동사]."""
    items: list[dict] = []
    fid = 0
    for v in SUBJ_VERBS:
        slots = v["themes"] if v["kind"] == "trans" else v["advs"]
        # 동사당 6 프레임(슬롯 전부 사용) → 최소쌍 6
        for slot in slots:
            s_hon = rng.choice(HON_SUBJ)
            s_pln = rng.choice(PLN_SUBJ)
            mid = obj_marker(slot) if v["kind"] == "trans" else slot
            frame = f"subj_f{fid:03d}"
            for cond, name, honored in (("honorific", s_hon, True), ("neutral", s_pln, False)):
                prompt = f"{subj_particle(name, honored)} {mid} "
                items.append({
                    "item_id": f"{frame}_{cond[:3]}",
                    "frame_id": frame,
                    "axis": "subject",
                    "morphology": "inflection",
                    "suppletive": False,
                    "cond": cond,
                    "expected": "honorific" if honored else "plain",
                    "prompt": prompt,
                    "honorific_target": v["hon"],
                    "plain_target": v["pln"],
                    "verb_lemma": v["lemma"],
                    "lexeme_id": f"subj:{_lexeme_id(v['lemma'])}",
                    "subject": name,
                    "subject_honored": int(honored),
                    "object": None,
                    "object_honored": None,
                    "object_case": None,
                    "control_flags": {"si_leak_blocked": True, "apjon_excluded": True,
                                      "animate_subject": True},
                })
            fid += 1
    return items


def build_object(rng: random.Random) -> list[dict]:
    """객체높임 보충법 최소쌍: [평어주어] [객체-께/-에게 또는 -을/-를] (목적물) [동사]."""
    items: list[dict] = []
    fid = 0
    for v in OBJ_VERBS:
        # dat 동사는 themes 6개, acc 동사는 객체 인물만(목적물 없음) → 6 프레임 균형
        n_frames = 6
        for k in range(n_frames):
            agent = rng.choice(PLN_AGENT)
            o_hon = rng.choice(HON_OBJ)
            o_pln = rng.choice(PLN_OBJ)
            frame = f"obj_f{fid:03d}"
            for cond, oname, honored in (("honorific", o_hon, True), ("neutral", o_pln, False)):
                if v["case"] == "dat":
                    theme = v["themes"][k % len(v["themes"])]
                    prompt = f"{agent} {dat_particle(oname, honored)} {obj_marker(theme)} "
                    obj_case = "dative"
                else:  # acc — 격조사에 높임 단서 없음, 보충법만이 신호
                    prompt = f"{agent} {acc_particle(oname)} "
                    obj_case = "accusative"
                items.append({
                    "item_id": f"{frame}_{cond[:3]}",
                    "frame_id": frame,
                    "axis": "object",
                    "morphology": "suppletion",
                    "suppletive": True,
                    "cond": cond,
                    "expected": "honorific" if honored else "plain",
                    "prompt": prompt,
                    "honorific_target": v["hon"],
                    "plain_target": v["pln"],
                    "verb_lemma": v["lemma_pln"],
                    "lexeme_id": f"obj:{_lexeme_id(v['lemma_pln'])}",
                    "subject": agent,
                    "subject_honored": 0,
                    "object": oname,
                    "object_honored": int(honored),
                    "object_case": obj_case,
                    "control_flags": {"si_leak_blocked": True, "apjon_excluded": True,
                                      "animate_subject": True, "plain_agent": True},
                })
            fid += 1
    return items


def selfcheck(items: list[dict]) -> None:
    """탐지기로 표면형 일관성 검증: 경어형은 탐지=1, 평형형은 탐지=0이어야."""
    errs = []
    for it in items:
        det = detect_subject_hon if it["axis"] == "subject" else detect_object_hon
        full_hon = it["prompt"] + it["honorific_target"]
        full_pln = it["prompt"] + it["plain_target"]
        if det(full_hon) != 1:
            errs.append(("HON_not_detected", it["item_id"], full_hon))
        if det(full_pln) != 0:
            errs.append(("PLN_falsely_detected", it["item_id"], full_pln))
    print(f"  자가검증: {len(items)}개 중 불일치 {len(errs)}건")
    for e in errs[:12]:
        print("   ✗", e[0], e[1], "|", e[2])
    if errs:
        print(f"   …총 {len(errs)}건 (탐지기·표면형 점검 필요)")


def main() -> None:
    rng = random.Random(SEED)
    subj = build_subject(rng)
    obj = build_object(rng)
    all_items = subj + obj

    out = ROOT / "data/stimuli/pairs.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in all_items) + "\n",
                   encoding="utf-8")

    n_subj_pairs = len({x["frame_id"] for x in subj})
    n_obj_pairs = len({x["frame_id"] for x in obj})
    n_subj_lex = len({x["lexeme_id"] for x in subj})
    n_obj_lex = len({x["lexeme_id"] for x in obj})
    print(f"주체높임(굴절): {len(subj)}자극 = {n_subj_pairs}최소쌍 × 2조건, "
          f"어휘 {n_subj_lex}개")
    print(f"객체높임(보충법): {len(obj)}자극 = {n_obj_pairs}최소쌍 × 2조건, "
          f"어휘 {n_obj_lex}개")
    print(f"총 {len(all_items)}자극 → {out.relative_to(ROOT)}")
    print("\n예시(주체 경어/중립):")
    for x in subj[:2]:
        print(f"  [{x['cond']:9}] {x['prompt']}「{x['honorific_target']} | {x['plain_target']}」")
    print("예시(객체 경어/중립):")
    for x in obj[:2]:
        print(f"  [{x['cond']:9}] {x['prompt']}「{x['honorific_target']} | {x['plain_target']}」")
    print()
    selfcheck(all_items)


if __name__ == "__main__":
    main()
