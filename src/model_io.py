"""모델 적재·후크 어댑터 (nnsight 기반 화이트박스 인프라).

연구계획서 §5.2: 주 도구 nnsight. EXAONE처럼 TransformerLens 미지원 구조에도
은닉 상태 후크를 건다. 세 모델의 잔차흐름 층 경로가 다르므로(EXAONE=transformer.h,
Llama/Qwen=model.layers) 어댑터로 추상화한다.

fp16(약 16GB) 또는 MPS. 24GB RAM에서 한 번에 한 모델만 적재(§5.2 주).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    layers_path: str   # 잔차흐름 디코더 블록 컨테이너 경로
    n_layers: int
    norm_path: str     # 최종 정규화(로짓 렌즈 unembed용)
    head_path: str     # 출력 임베딩(lm_head)
    trust_remote_code: bool = False


SPECS: dict[str, ModelSpec] = {
    "exaone": ModelSpec("exaone", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                        "transformer.h", 32, "transformer.ln_f", "lm_head",
                        trust_remote_code=True),
    "llama": ModelSpec("llama", "NousResearch/Meta-Llama-3.1-8B-Instruct",
                       "model.layers", 32, "model.norm", "lm_head"),
    "qwen": ModelSpec("qwen", "Qwen/Qwen3-8B", "model.layers", 36,
                      "model.norm", "lm_head"),
    # 크기 사다리(소형)
    "exaone24": ModelSpec("exaone24", "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
                          "transformer.h", 30, "transformer.ln_f", "lm_head",
                          trust_remote_code=True),
    "qwen17": ModelSpec("qwen17", "Qwen/Qwen3-1.7B", "model.layers", 28,
                        "model.norm", "lm_head"),
}


def _resolve(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def load_model(key: str, dtype="float16", device_map="mps"):
    """nnsight LanguageModel 적재. 반환: (lm, spec)."""
    import torch
    from nnsight import LanguageModel

    spec = SPECS[key]
    torch_dtype = getattr(torch, dtype)
    lm = LanguageModel(
        spec.repo,
        torch_dtype=torch_dtype,
        trust_remote_code=spec.trust_remote_code,
        device_map=device_map,
        dispatch=True,
    )
    return lm, spec


def layer_container(lm, spec: ModelSpec):
    """nnsight envoy로 디코더 블록 컨테이너(layers/h) 반환."""
    return _resolve(lm, spec.layers_path)


def residual_out(lm, spec: ModelSpec, layer: int):
    """layer번째 블록의 잔차흐름 출력 텐서(hidden state). 블록 output은 보통 튜플."""
    blocks = layer_container(lm, spec)
    return blocks[layer].output[0]
