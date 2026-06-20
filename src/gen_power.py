"""생성 검정력 (n↑ + 다중 샘플 + 부트스트랩 CI). 행동 수준 해리 검정.

기존 생성 검증(n=12 greedy)은 검정력 부족. 여기서는 v3 3범주(subject 굴절·subject_suppl 주체
보충법·object 객체 보충법)에서 honorific 항목마다 K개 샘플(temperature)을 생성해 Kiwi 경어
산출률을 구하고, 항목 단위 부트스트랩 95% CI를 보고한다. 핵심: 주체 보충법도 객체처럼 생성
실패하면 → 실패는 보충법(어휘 인출), 굴절처럼 성공하면 → 객체 역할.

실행: STIM=v3 .venv/bin/python -m src.gen_power <model> [test]
산출: analysis/output/genpower_<key><suf>.json
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

from src.honor_axes import detect_object_hon, detect_subject_hon  # noqa: E402
from src.model_io import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
K = 6           # 항목당 샘플 수
TEMP = 0.8


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    test = len(sys.argv) > 2 and sys.argv[2] == "test"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = f"data/stimuli/pairs_{stim}.jsonl" if stim else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]

    lm, spec = load_model(key)
    tok = lm.tokenizer
    axes = sorted({it["axis"] for it in items})
    res = {"model": key, "stim": stim or "v1", "K": K, "temp": TEMP}
    rng = np.random.default_rng(0)

    for ax in axes:
        det = detect_object_hon if ax == "object" else detect_subject_hon
        hon = [it for it in items if it["axis"] == ax and it["cond"] == "honorific"]
        if test:
            hon = hon[:3]
        elif len(hon) > 24:
            hon = [hon[j] for j in rng.permutation(len(hon))[:24]]
        item_rates = []
        for it in hon:
            ids = tok(it["prompt"], return_tensors="pt")["input_ids"]
            hits = 0
            for _ in range(K):
                with lm.generate(ids, max_new_tokens=6, do_sample=True, temperature=TEMP, top_p=0.95):
                    g = lm.generator.output.save()
                out = g if isinstance(g, torch.Tensor) else g.value
                text = tok.decode(out[0], skip_special_tokens=True)
                hits += det(text)
            item_rates.append(hits / K)
        item_rates = np.array(item_rates)
        # 항목 단위 부트스트랩 CI
        boots = [item_rates[rng.integers(0, len(item_rates), len(item_rates))].mean()
                 for _ in range(2000)] if len(item_rates) else [0]
        res[ax] = {"n_items": len(hon), "honorific_gen_rate": round(float(item_rates.mean()), 3),
                   "ci_lo": round(float(np.percentile(boots, 2.5)), 3),
                   "ci_hi": round(float(np.percentile(boots, 97.5)), 3)}
        print(f"[{key}/{ax}] 경어 생성률 {res[ax]['honorific_gen_rate']:.3f} "
              f"CI[{res[ax]['ci_lo']:.3f},{res[ax]['ci_hi']:.3f}] (n={len(hon)}×{K}샘플)", flush=True)

    if test:
        print("test OK", flush=True)
        return
    outp = ROOT / f"analysis/output/genpower_{key}{suf}.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
