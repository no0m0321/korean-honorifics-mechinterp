"""3단계 활성값 패칭 (가설 M3): 인과 추적.

clean(경어 촉발) 실행의 잔차를 corrupt(중립) 실행에 층별로 치환해, 출력이 평형→경어로
얼마나 회복되는지(logit-diff 회복률) 측정한다. 최소쌍은 높임 논항(상류)만 다르고 동사·부사어
(하류)는 같으므로, 각 실행의 '결정 지점'(분기 직전 마지막 위치)에서 잔차를 치환한다.

회복률_L = (patched_LD − corrupt_LD) / (clean_LD − corrupt_LD)
  LD = logit(경어형 첫토큰) − logit(평형형 첫토큰). 1.0=완전회복, 0=무효.
예측(M3): -시-(굴절)는 중기 층 치환으로 회복되나, 보충법은 회복이 더 어렵고 덜(검색 병목).

검증: STIM=v2 .venv/bin/python -m src.patch <model> test
실행: STIM=v2 .venv/bin/python -m src.patch <model>
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.align import align_item  # noqa: E402
from src.model_io import layer_container, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _ld(logits, hon_first, pln_first):
    lg = logits[0, -1].float()
    return float(lg[hon_first] - lg[pln_first])


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    test = len(sys.argv) > 2 and sys.argv[2] == "test"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = "data/stimuli/pairs_v2.jsonl" if stim == "v2" else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]

    # frame별 clean(honorific)+corrupt(neutral) 짝
    byframe = defaultdict(dict)
    for it in items:
        byframe[it["frame_id"]][it["cond"]] = it
    frames = [(f, d["honorific"], d["neutral"]) for f, d in byframe.items()
              if "honorific" in d and "neutral" in d]
    if test:
        frames = [next(f for f in frames if f[1]["axis"] == "subject"),
                  next(f for f in frames if f[1]["axis"] == "object")]

    lm, spec = load_model(key)
    tok = lm.tokenizer
    bos = tok.bos_token_id
    nL = spec.n_layers
    print(f"[{key}] frames={len(frames)} layers={nL}", flush=True)

    recov = {"subject": np.zeros((0, nL)), "object": np.zeros((0, nL))}
    rs = {"subject": [], "object": []}
    for fi, (frame, clean_it, corr_it) in enumerate(frames):
        a_c = align_item(tok, clean_it["prompt"], clean_it["honorific_target"], clean_it["plain_target"])
        a_k = align_item(tok, corr_it["prompt"], corr_it["honorific_target"], corr_it["plain_target"])
        if a_c.hon_first is None or a_c.hon_first != a_k.hon_first or a_c.pln_first != a_k.pln_first:
            continue  # 분기 토큰 불일치 쌍 제외
        hf, pf = a_c.hon_first, a_c.pln_first
        clean_ids = torch.tensor([([bos] if bos is not None else []) + a_c.prefix_ids])
        corr_ids = torch.tensor([([bos] if bos is not None else []) + a_k.prefix_ids])

        # clean: 모든 층 마지막위치 잔차 + logits
        clean_res = []
        with lm.trace(clean_ids):
            blocks = layer_container(lm, spec)
            for L in range(nL):
                clean_res.append(blocks[L].output[0].save())
            clean_logits = lm.output.logits.save()
        clean_vec = []
        for L in range(nL):
            t = clean_res[L]
            t = t if isinstance(t, torch.Tensor) else t.value
            clean_vec.append((t[-1] if t.ndim == 2 else t[0, -1]).detach())
        clean_LD = _ld(clean_logits if isinstance(clean_logits, torch.Tensor) else clean_logits.value, hf, pf)

        # corrupt baseline
        with lm.trace(corr_ids):
            corr_logits = lm.output.logits.save()
        corr_LD = _ld(corr_logits if isinstance(corr_logits, torch.Tensor) else corr_logits.value, hf, pf)

        denom = clean_LD - corr_LD
        if abs(denom) < 1e-3:
            continue
        # 층별 패칭
        row = np.zeros(nL)
        for L in range(nL):
            with lm.trace(corr_ids):
                blocks = layer_container(lm, spec)
                out = blocks[L].output[0]
                if out.ndim == 2:
                    out[-1, :] = clean_vec[L]
                else:
                    out[0, -1, :] = clean_vec[L]
                pl = lm.output.logits.save()
            patched_LD = _ld(pl if isinstance(pl, torch.Tensor) else pl.value, hf, pf)
            row[L] = (patched_LD - corr_LD) / denom
        ax = clean_it["axis"]
        rs[ax].append(row)
        if test:
            print(f"  [{ax}] clean_LD={clean_LD:+.2f} corr_LD={corr_LD:+.2f} "
                  f"회복 정점 L{int(np.argmax(row))}={row.max():.2f} 최종L{nL-1}={row[-1]:.2f}", flush=True)
        elif (fi + 1) % 20 == 0:
            print(f"  {fi+1}/{len(frames)}", flush=True)

    if test:
        print("test OK", flush=True)
        return
    out = {"model": key, "stim": stim or "v1", "n_layers": nL}
    for ax in ("subject", "object"):
        arr = np.array(rs[ax])
        out[ax] = {"n": len(arr), "mean_recovery_by_layer": np.nanmean(arr, axis=0).round(3).tolist(),
                   "peak_layer": int(np.nanmean(arr, axis=0).argmax()) if len(arr) else None,
                   "peak_recovery": float(np.nanmean(arr, axis=0).max()) if len(arr) else None}
    outp = ROOT / f"analysis/output/patch_{key}{suf}.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for ax in ("subject", "object"):
        print(f"[{key}/{ax}] n={out[ax]['n']} 정점 L{out[ax]['peak_layer']} "
              f"회복={out[ax]['peak_recovery']}", flush=True)
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
