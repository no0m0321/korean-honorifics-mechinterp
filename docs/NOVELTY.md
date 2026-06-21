# 선행 연구 검색 및 차별성 (2026-06, 웹 검색 기반)

## 동일·유사 선행 연구 없음 — '한국어 경어 보충법의 기계적 해석'은 최초

### 가장 가까운 한국어 경어×LLM 연구 (그러나 본 연구와 다름)
- **How Language Models Understand Honorific Mismatches in Korean** (Language Research 2024,
  doi:10.30961/lr.2024.60.3.303): surprisal(블랙박스)를 인코더 모델(KR-BERT·KoELECTRA·
  KLUE-RoBERTa)에 적용, 경어 불일치(YN/NY) 수용성 판단. → 본 연구는 (i) 생성 LLM, (ii) 기계적
  해석(프로빙·렌즈·패칭·조향), (iii) 보충법 *생성 실패의 기전*으로 차별.
- **Pragmatic Competence Evaluation of LLMs for Korean** (PACLIC 2024, arXiv:2403.12675):
  화용 능력 행동 평가. → 기계적 해석 아님.
- **KITE** (arXiv:2510.15558): 한국어 지시따르기 벤치마크. → 행동 평가.

### 본 연구의 기전이 연결되는 일반 해석학 문헌 (한국어 경어/보충법엔 미적용)
- **From Early Encoding to Late Suppression** (arXiv:2604.00778, character counting): '프로브엔
  정보 있으나 후기층 억제'. → 본 연구의 untuned 중간층 신호와 구조 유사하나, **튜닝 렌즈 검증으로
  인공물임을 밝혀 철회**(방법적 엄밀성).
- **When Transformers Know but Don't Tell** (arXiv:2406.14673): 앎-행함 간극의 장문맥 사례.
- **Promote, Suppress, Iterate** (arXiv:2502.20475): 사실 질의의 promote/suppress 기전.
- **Emergent Specialization: Rare Token Neurons** (arXiv:2505.12822): 빈도 비례 토큰 조절 뉴런.
- **Whether, Not Which** (arXiv:2603.22295): 기계적 해석으로 두 처리과정 해리(방법 구조 유사).
- 활성값 조향: Rimsky CAA(arXiv:2312.06681), Turner ActAdd(2308.10248), Jorgensen mean-centring
  (2312.03813).

## 신규 기여 (요약)
1. 한국어 경어 보충법의 **최초** 기계적 해석(생성 LLM, 4기법 삼각측량).
2. 굴절–보충법을 **합성적 자질 부착 vs 항목별 어휘 인출**의 자연 실험으로 활용.
3. 보충법 생성 실패를 **첫 분기 토큰 commitment**으로 인과 국소화(강제 디코딩·생성 검정력).
4. v3 분리 실험으로 **suppletion(주효과) vs object(추가 패널티)** 이중구조 입증.
5. 기전의 **규모·계열 의존성**(소형/EXAONE 진성 결손 → 대형/영어 commitment 병목).
6. 적대 검증 2회·튜닝 렌즈로 과해석 정정('정도의 비대칭').
