"""강제 디코딩으로 commitment 병목 인과 확증 (핵심 기전).

가설: Llama의 객체 실패는 '첫 토큰 commitment' 병목(보충법을 알지만 첫 분기 토큰에서 평형에
commit), EXAONE은 진성 표상 실패. 검정: 객체 honorific 항목에서 **첫 보충법 토큰을 강제**한 뒤
greedy 생성 → 보충법을 완성하는가(Kiwi 객체높임 탐지).
예측: Llama는 강제 시 완성률 급증(병목 우회), EXAONE은 덜 증가(표상 결손이라 강제해도 안 됨).

세 조건: (a) baseline 자유생성, (b) 첫 경어토큰 강제, (c) 첫 평형토큰 강제(통제).
실행: STIM=v2 .venv/bin/python -m src.forced_decode <model> [test]
산출: analysis/output/forced_<key><suf>.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import torch  # noqa: E402

from src.align import align_item  # noqa: E402
from src.honor_axes import detect_object_hon, detect_subject_hon  # noqa: E402
from src.model_io import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    test = len(sys.argv) > 2 and sys.argv[2] == "test"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = "data/stimuli/pairs_v2.jsonl" if stim == "v2" else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]

    lm, spec = load_model(key)
    tok = lm.tokenizer

    def gen_detect(prompt, force_ids, det, n_new=5):
        """prompt(+강제토큰들) 뒤 greedy 생성 후 탐지기 적용."""
        base = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
        ids = torch.tensor([base + list(force_ids)])
        with lm.generate(ids, max_new_tokens=n_new, do_sample=False):
            g = lm.generator.output.save()
        out = g if isinstance(g, torch.Tensor) else g.value
        text = tok.decode(out[0], skip_special_tokens=True)
        return det(text), text

    res = {"model": key, "stim": stim or "v1"}
    for ax in ("object", "subject"):
        det = detect_object_hon if ax == "object" else detect_subject_hon
        hon_items = [it for it in items if it["axis"] == ax and it["cond"] == "honorific"]
        if test:
            hon_items = hon_items[:4]
        base_rate = forced_hon = forced_pln = n = 0
        for it in hon_items:
            a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
            if a.hon_first is None:
                continue
            b, _ = gen_detect(it["prompt"], [], det)
            fh, txt = gen_detect(it["prompt"], [a.hon_first], det)
            fp, _ = gen_detect(it["prompt"], [a.pln_first], det)
            base_rate += b; forced_hon += fh; forced_pln += fp; n += 1
            if test:
                print(f"  [{ax}] base={b} 강제경어첫토큰={fh} 강제평형={fp} | '{txt[-12:]}'", flush=True)
        res[ax] = {"n": n,
                   "baseline_honorific_rate": round(base_rate / max(n, 1), 3),
                   "forced_hon_first_rate": round(forced_hon / max(n, 1), 3),
                   "forced_pln_first_rate": round(forced_pln / max(n, 1), 3),
                   "rescue": round((forced_hon - base_rate) / max(n, 1), 3)}
        print(f"[{key}/{ax}] baseline={res[ax]['baseline_honorific_rate']} "
              f"강제경어첫토큰={res[ax]['forced_hon_first_rate']} (구제 {res[ax]['rescue']:+.3f}) "
              f"강제평형통제={res[ax]['forced_pln_first_rate']} n={n}", flush=True)

    if test:
        print("test OK", flush=True)
        return
    outp = ROOT / f"analysis/output/forced_{key}{suf}.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
