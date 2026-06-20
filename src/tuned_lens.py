"""튜닝 렌즈 (사전등록 충실도, 메모리 효율판).

이전판은 모든 층 잔차를 RAM에 모아 thrash(swap)했다. 여기서는 잔차를 저장하지 않고 층별
XᵀX·XᵀY 를 **증분 누적**해 ridge를 푼다(메모리 O(층×H²), 모델과 공존). 충분한 학습 위치로
제대로 학습. 튜닝 렌즈 logit_L = unembed(A_L·h_L + b_L). 자극 객체/주체 첫토큰 LD 궤적을
untuned vs tuned로 비교해 'untuned 중간층 보충법 선호'가 인공물인지 검증.

실행: STIM=v2 .venv/bin/python -m src.tuned_lens <model>
산출: analysis/output/tunedlens_<key><suf>.json
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

from src.align import align_item  # noqa: E402
from src.model_io import layer_container, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAXPOS = 8000
ALPHA = 1000.0


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "exaone"
    stim = os.environ.get("STIM", "")
    suf = f"_{stim}" if stim else ""
    pairs_f = f"data/stimuli/pairs_{stim}.jsonl" if stim else "data/stimuli/pairs.jsonl"
    items = [json.loads(l) for l in (ROOT / pairs_f).read_text().splitlines()]

    lm, spec = load_model(key)
    tok = lm.tokenizer
    bos = tok.bos_token_id
    BOS = [bos] if bos is not None else []
    base = getattr(lm, "_model", None) or getattr(lm, "model", None)
    norm_mod = base
    for p in spec.norm_path.split("."):
        norm_mod = getattr(norm_mod, p)
    head_mod = base.lm_head
    dev, dt = head_mod.weight.device, head_mod.weight.dtype
    nL = spec.n_layers

    from datasets import load_dataset
    ds = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")

    H = None
    XtX = XtY = None  # 층별 [H,H] 누적
    pos = 0
    n_sent = 0
    for i in range(len(ds)):
        text = (ds[i].get("document") or "")[:200]
        if len(text) < 20:
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=48)["input_ids"]
        saved = []
        with lm.trace(ids):
            blk = layer_container(lm, spec)
            for L in range(nL):
                saved.append(blk[L].output[0].save())
        mats = []
        for L in range(nL):
            t = saved[L]; t = t if isinstance(t, torch.Tensor) else t.value
            mats.append((t if t.ndim == 2 else t[0]).detach().float().cpu().numpy())  # [seq,H]
        if XtX is None:
            H = mats[0].shape[-1]
            XtX = [np.zeros((H, H), dtype=np.float64) for _ in range(nL)]
            XtY = [np.zeros((H, H), dtype=np.float64) for _ in range(nL)]
        hf = mats[nL - 1]  # 최종층
        for L in range(nL):
            x = mats[L]
            XtX[L] += x.T @ x
            XtY[L] += x.T @ hf
        pos += mats[0].shape[0]; n_sent += 1
        if pos >= MAXPOS:
            break
    print(f"[{key}] 학습 위치 {pos} ({n_sent}문장), H={H}", flush=True)

    # 층별 ridge 해: (XtX + αI)^-1 XtY
    translators = []
    eye = np.eye(H) * ALPHA
    for L in range(nL):
        W = np.linalg.solve(XtX[L] + eye, XtY[L])  # [H,H]
        translators.append(torch.tensor(W, device=dev, dtype=dt))
        XtX[L] = XtY[L] = None

    def proj(h, L, tuned):
        ht = h.to(dev, dt)
        if tuned:
            ht = ht @ translators[L]
        with torch.no_grad():
            return head_mod(norm_mod(ht))

    res = {"model": key, "stim": stim or "v1", "n_layers": nL, "train_pos": int(pos)}
    for ax in ("object", "subject"):
        hon = [it for it in items if it["axis"] == ax and it["cond"] == "honorific"]
        unt = np.zeros((len(hon), nL)); tun = np.zeros((len(hon), nL)); ok = 0
        for it in hon:
            a = align_item(tok, it["prompt"], it["honorific_target"], it["plain_target"])
            if a.hon_first is None:
                continue
            ids = torch.tensor([BOS + a.prefix_ids])
            saved = []
            with lm.trace(ids):
                blk = layer_container(lm, spec)
                for L in range(nL):
                    saved.append(blk[L].output[0].save())
            for L in range(nL):
                t = saved[L]; t = t if isinstance(t, torch.Tensor) else t.value
                h = (t[-1] if t.ndim == 2 else t[0, -1]).detach()
                lu = proj(h, L, False); lt = proj(h, L, True)
                unt[ok, L] = float(lu[a.hon_first] - lu[a.pln_first])
                tun[ok, L] = float(lt[a.hon_first] - lt[a.pln_first])
            ok += 1
        unt, tun = unt[:ok], tun[:ok]
        um, tm = np.nanmean(unt, 0), np.nanmean(tun, 0)
        res[ax] = {"n": ok,
                   "untuned_mean_by_layer": um.round(2).tolist(),
                   "tuned_mean_by_layer": tm.round(2).tolist(),
                   "untuned_mid_max": round(float(um[nL // 4:3 * nL // 4].max()), 2),
                   "tuned_mid_max": round(float(tm[nL // 4:3 * nL // 4].max()), 2),
                   "untuned_final": round(float(um[-1]), 2), "tuned_final": round(float(tm[-1]), 2)}
        print(f"[{key}/{ax}] 중간층최대 untuned={res[ax]['untuned_mid_max']}→tuned={res[ax]['tuned_mid_max']} "
              f"| 최종 untuned={res[ax]['untuned_final']}→tuned={res[ax]['tuned_final']}", flush=True)

    outp = ROOT / f"analysis/output/tunedlens_{key}{suf}.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", outp.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
