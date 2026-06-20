"""M2 시퀀스 로그우도 버전 (첫 분기 토큰의 비대칭 보정).

기존 cache_lens는 경어형/평형형의 '첫 분기 토큰' logit-diff만 본다. 그러나 주체는 어미 분기·
객체는 어간 분기이고, 보충법 경어형(말씀드렸다·뵈었다)은 다토큰이라 첫 토큰이 전체 형태 선호를
대표 못할 수 있다. 여기서는 분기 이후 **동사 형태 전체의 시퀀스 로그우도 차**를 측정한다:
  seq_LL_diff = logΣp(경어형 동사토큰들 | prefix) − logΣp(평형형 동사토큰들 | prefix)
align의 hon_ids/pln_ids(분기 이후 토큰들) 사용. 다토큰 형태를 공정하게 비교.

실행: STIM=v2 .venv/bin/python -m src.seq_m2 <model> [test]
산출: analysis/output/seqm2_<key><suf>.npz, _summary.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.align import align_item  # noqa: E402
from src.model_io import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    test = len(sys.argv) > 2 and sys.argv[2] == "test"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = f"data/stimuli/pairs_{stim}.jsonl" if stim else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]
    if test:
        items = ([it for it in items if it["axis"] == "subject"][:3]
                 + [it for it in items if it["axis"] == "object"][:3])

    lm, spec = load_model(key)
    tok = lm.tokenizer
    bos = tok.bos_token_id
    BOS = [bos] if bos is not None else []

    def seq_ll(prefix_ids, cont_ids):
        """prefix 다음 cont_ids 토큰들의 로그확률 합."""
        full = torch.tensor([BOS + prefix_ids + cont_ids])
        with lm.trace(full):
            logits = lm.output.logits.save()
        lg = (logits if isinstance(logits, torch.Tensor) else logits.value)[0].float()
        lp = F.log_softmax(lg, dim=-1)
        start = len(BOS) + len(prefix_ids) - 1
        return float(sum(lp[start + i, cont_ids[i]] for i in range(len(cont_ids))))

    diff = np.full(len(items), np.nan, dtype=np.float32)
    meta = []
    for i, it in enumerate(items):
        a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
        if not a.hon_ids or not a.pln_ids:
            meta.append(it); continue
        ll_h = seq_ll(a.prefix_ids, a.hon_ids)
        ll_p = seq_ll(a.prefix_ids, a.pln_ids)
        # 길이 정규화(토큰당 평균 로그우도 차) — 다토큰 공정비교
        diff[i] = (ll_h / len(a.hon_ids)) - (ll_p / len(a.pln_ids))
        meta.append(it)
        if test:
            print(f"  [{it['axis']:7}/{it['cond']:9}] seqLL_diff(토큰평균)={diff[i]:+.3f} "
                  f"hon_tok={len(a.hon_ids)} pln_tok={len(a.pln_ids)}", flush=True)

    if test:
        print("test OK", flush=True)
        return

    axis = np.array([m["axis"] for m in meta])
    cond = np.array([m["cond"] for m in meta])
    lex = np.array([m["lexeme_id"] for m in meta])
    np.savez_compressed(ROOT / f"analysis/output/seqm2_{key}{suf}.npz",
                        diff=diff, axis=axis, cond=cond, lexeme=lex)
    # 요약: honorific 조건 축별 평균 + 추월(>0) 비율
    summ = {"model": key, "stim": stim or "v1", "metric": "token-mean seq logprob diff (hon-pln)"}
    hon = cond == "honorific"
    for ax in ("subject", "object"):
        m = hon & (axis == ax) & np.isfinite(diff)
        summ[ax] = {"mean": round(float(np.mean(diff[m])), 3),
                    "frac_positive": round(float((diff[m] > 0).mean()), 3), "n": int(m.sum())}
    (ROOT / f"analysis/output/seqm2_{key}{suf}_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=2), flush=True)
    print("저장: seqm2", flush=True)


if __name__ == "__main__":
    main()
