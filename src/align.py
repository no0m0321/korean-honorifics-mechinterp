"""토크나이저 분기 토큰 정렬 (계획서 §4.2 주: '동사 위치' 표준화).

한국어 형태소가 BPE 토큰 경계와 어긋날 수 있으므로, 경어형/평형형 표면형이 처음
갈라지는 분기 토큰을 토큰 단위 최장공통접두(LCP)로 찾는다. 이 방식은 두 축을 통일한다:
  - 주체높임(굴절): 어간 '읽' 공유 후 어미에서 분기 (읽으셨다 / 읽었다)
  - 객체높임(보충법): 어간 자체가 분기 (드렸다 / 주었다)

측정에 쓰이는 산출:
  - prefix_ids: 분기 직전까지의 토큰(프롬프트 + 공유 어간). 다음 토큰 분포를 보는 위치는
    이 마지막 토큰의 출력이다.
  - hon_first / pln_first: 경어형·평형형의 첫 분기 토큰 ID (로짓 차 logit-diff용)
  - hon_ids / pln_ids: 분기 이후 동사 잔여 토큰(시퀀스 로그확률 비교용)

instruct 모델이지만 형태론 측정을 깨끗이 하기 위해 chat template 없이 raw 완성 방식을
쓴다(add_special_tokens는 호출부에서 BOS만 선택적으로 부여).

가중치 불필요(토크나이저만). 검증: python -m src.align <model_key>
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Alignment:
    prefix_ids: list[int]   # 분기 직전까지(프롬프트 + 공유 어간 토큰)
    branch_pos: int         # 분기 토큰 인덱스 (= len(prefix_ids))
    hon_first: int | None   # 경어형 첫 분기 토큰 ID
    pln_first: int | None   # 평형형 첫 분기 토큰 ID
    hon_ids: list[int]      # 분기 이후 경어형 잔여 토큰
    pln_ids: list[int]      # 분기 이후 평형형 잔여 토큰
    n_prompt_tok: int       # 순수 프롬프트 토큰 수(공유 어간 제외)


def align_item(tokenizer, prompt: str, hon: str, pln: str) -> Alignment:
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    ph, pp = enc(prompt + hon), enc(prompt + pln)
    i = 0
    while i < len(ph) and i < len(pp) and ph[i] == pp[i]:
        i += 1
    return Alignment(
        prefix_ids=ph[:i],
        branch_pos=i,
        hon_first=ph[i] if i < len(ph) else None,
        pln_first=pp[i] if i < len(pp) else None,
        hon_ids=ph[i:],
        pln_ids=pp[i:],
        n_prompt_tok=len(enc(prompt)),
    )


def is_valid(a: Alignment) -> bool:
    """분기 토큰이 잘 정의되었는가(둘 다 존재하며 서로 다름)."""
    return (a.hon_first is not None and a.pln_first is not None
            and a.hon_first != a.pln_first)


if __name__ == "__main__":
    import json
    import sys
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer

    from src.model_io import SPECS

    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    spec = SPECS[key]
    tok = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=spec.trust_remote_code)
    root = Path(__file__).resolve().parent.parent
    items = [json.loads(l) for l in (root / "data/stimuli/pairs.jsonl").read_text().splitlines()]

    bad, by_axis = [], {}
    for it in items:
        a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
        rec = by_axis.setdefault(it["axis"], {"n": 0, "branch_off": 0, "hon_tok": 0, "pln_tok": 0})
        rec["n"] += 1
        rec["branch_off"] += a.branch_pos - a.n_prompt_tok  # 공유 어간 토큰 수
        rec["hon_tok"] += len(a.hon_ids)
        rec["pln_tok"] += len(a.pln_ids)
        if not is_valid(a):
            bad.append((it["item_id"], it["prompt"], it["honorific_target"], it["plain_target"]))

    print(f"=== {key} ({spec.repo}) 토큰 정렬 ===")
    for axis, r in by_axis.items():
        n = r["n"]
        print(f"  {axis:8} n={n}  공유어간토큰 평균={r['branch_off']/n:.2f}  "
              f"경어형 동사토큰 평균={r['hon_tok']/n:.2f}  평형형 평균={r['pln_tok']/n:.2f}")
    print(f"  분기 미정의(무효) 항목: {len(bad)}건")
    for b in bad[:8]:
        print("   ✗", b)

    # 예시
    print("  예시:")
    for it in (items[0], items[192]):
        a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
        print(f"   [{it['axis']}] {it['prompt']}「{it['honorific_target']}|{it['plain_target']}」 "
              f"→ 분기위치{a.branch_pos} hon_first={a.hon_first}({tok.decode([a.hon_first])!r}) "
              f"pln_first={a.pln_first}({tok.decode([a.pln_first])!r})")
