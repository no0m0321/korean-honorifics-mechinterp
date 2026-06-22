# 한국어 객체 높임 보충법 실패의 기계적 해석

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20791500.svg)](https://doi.org/10.5281/zenodo.20791500)

> 굴절–보충법 비대칭은 모델 내부 어디에서 생기는가
> *A Mechanistic Account of Suppletive Object-Honorific Failure in Korean LLMs*
>
> 김승우 (잠신고등학교) · [논문 초고](docs/PAPER_DRAFT.md) · [결과 정직본](RESULTS.md) · [사전등록](PREREGISTRATION.md)

선행 연구 김승우(2026a) 「한국어 경어법에 대한 LLM의 프롬프트 민감성」의 직접 후속.
블랙박스 행동 관찰 → 화이트박스 기계적 해석으로의 전환.

## 핵심 질문

세 LLM 모두 생성적 굴절 형태소 주체 높임 `-시-`는 능숙하게(43–94%) 다루는 반면, 어휘
특정적 객체 높임 보충법(드리다·여쭙다)은 거의 생성하지 못한다(0–32%). 이 형태론적
**해리**가 모델 내부에서 **‘경어가 필요함을 모르는(인코딩) 실패’**인지 아니면 **‘알지만
올바른 어휘를 꺼내지 못하는(검색) 실패’**인지를 인과적으로 규명한다.

## 방법: 난이도순 4단계 파이프라인

| 단계 | 기법 | 묻는 것 | 가설 |
|---|---|---|---|
| 1 | 프로빙 (선형 프로브 + 통제 과제) | 객체 높임 요구가 인코딩되는가? | M1, M6 |
| 2 | 로짓/튜닝된 렌즈 | 올바른 형태는 어느 층에서 우세해지는가? | M2 |
| 3 | 활성값 패칭 (인과 추적) | 어떤 회로가 경어 신호를 운반하는가? | M3 |
| 4 | CAA 조향 (인과 수정) | 전역 벡터로 `-시-`는 고치고 보충법은 못 고치는가? | M4 |

각 단계는 그 자체로 독립 논문이 되도록 설계(위험 분산). 모든 결론은 네 기법의
삼각측량 일치가 성립할 때만 채택한다.

## 대상 모델

- EXAONE-3.5-7.8B-Instruct (32층, 한국어 특화) — `transformer.h`
- Llama-3.1-8B-Instruct (32층, 영어 중심) — `model.layers`
- Qwen3-8B (36층, 다국어) — `model.layers`

전부 hidden 4096, fp16 적재. nnsight 0.7로 은닉 상태 후크.

## 디렉터리

```
src/
  make_stimuli.py   대조 최소쌍 자극 생성 (계획서 §4)
  honor_axes.py     Kiwi 기반 -시-/보충법 탐지기 (2026a 재사용·-시- 융합형 보강)
  model_io.py       nnsight 모델 적재·층 후크 어댑터
  align.py          토크나이저 분기 토큰 정렬 (Kiwi 동사 위치 표준화)
  cache_acts.py     자극 잔차흐름 활성값 캐싱
  probe.py          1단계 선형 프로빙 + 통제 과제 + MDL
data/
  stimuli/pairs.jsonl   자극 (264, seed=42)
  freq/                 말뭉치 로그빈도 (M5)
analysis/               단계별 분석·그림
_smoke/                 인프라 실현가능성 검증
PREREGISTRATION.md      사전등록 (데이터 수집 전 동결)
```

## 실행

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
.venv/bin/python -m src.make_stimuli       # 자극 생성
.venv/bin/python _smoke/smoke_nnsight.py exaone   # 후크 검증
```

## 상태

진행 중. 인프라·자극 설계·사전등록 완료. 1단계 프로빙 구현 단계.
