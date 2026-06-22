# 공개 진행 상황 및 남은 절차

## ✅ 완료
- **GitHub 공개**: https://github.com/no0m0321/korean-honorifics-mechinterp (PUBLIC)
  - 코드·자극·결과(RESULTS.md)·사전등록(PREREGISTRATION.md)·논문 초고(docs/PAPER_DRAFT.md)·docx·그림
- **릴리스 v1.0**: https://github.com/no0m0321/korean-honorifics-mechinterp/releases/tag/v1.0 (논문 docx 첨부)

## 🔲 Zenodo DOI (선행연구와 동일 경로 — 브라우저 필요)
1. https://zenodo.org 접속 → GitHub로 로그인(OAuth).
2. 상단 메뉴 → **GitHub** → 저장소 목록에서 `korean-honorifics-mechinterp` 토글을 **ON**.
3. (토글 ON 후) GitHub에서 **새 릴리스** 생성 → Zenodo가 자동 아카이브 → DOI 발급.
   - 토글이 v1.0 생성 전이었으므로, 새 릴리스 v1.0.1을 만들어야 Zenodo가 잡습니다.
   - 원하시면 토글 ON 확인 후 제가 `gh release create v1.0.1`을 실행해 드립니다.
4. DOI 발급 후 메타데이터에서 저자 'Kim Seungwoo / Jamsin High School' 확인·수정, README에 DOI 배지.

## 🔲 OSF (브라우저 로그인 필요)
1. https://osf.io 로그인 → 새 프로젝트 생성(예: "Korean Honorific Suppletion Mechanistic Interp").
2. **Add-ons → GitHub** 연결(OAuth) → `korean-honorifics-mechinterp` 선택 → 전체 파일 접근.
3. 또는 OSF Storage에 docx·PDF 직접 업로드(파일 공유 폴더 제약 있음 — 선행연구 메모 참조).
4. 공개 전환 + (원하면) 정식 Registration으로 동결+DOI.

## 🔲 arXiv (제약 있음)
- arXiv는 신규 제출자에게 **endorsement(추천)**를 요구합니다(특히 cs.CL). 무소속 고교생은 추천인이
  필요할 수 있습니다. 대안: OSF Preprints, 또는 지도교사·연구자 추천 확보 후 제출.
- 제출 형식: PDF(docx→PDF 변환) 또는 LaTeX. 분야: cs.CL(Computation and Language).

## 참고
- 선행연구(korean-honorifics-llm)는 GitHub + OSF + Zenodo DOI(10.5281/zenodo...) 모두 공개돼 있어
  동일 절차를 그대로 따르면 됩니다.
