# Server2-rag: 랭그래프 기반 에이전틱 RAG 시스템

회의 텍스트를 입력 받아 LLM과 벡터 검색을 활용한 에이전틱 RAG(Retrieval-Augmented Generation) 시스템입니다. 텍스트를 의미 단위로 분할하고, Google Gemini를 활용해 요약·쿼리 생성, 문서 검색 및 적합성 평가를 수행하는 종합 파이프라인을 제공합니다. 

## 주요 개선 사항 (2025-05-22)

### 1. 향상된 문서 관련성 평가
- 문서 관련성 평가 로직 개선
- 재시도 시 피드백 기반 쿼리 개선
- 문서 메타데이터 활용 강화

### 2. 강화된 재시도 메커니즘
- 최대 재시도 횟수 설정 가능
- 피드백 기반 쿼리 개선
- 재시도 시 이전 컨텍스트 유지

### 3. 개선된 결과 처리
- 상세 실행 통계 추가
- 오류 처리 및 로깅 강화
- 처리 상태 추적 기능 개선

### 4. 성능 모니터링
- 함수별 실행 시간 측정
- 리소스 사용량 모니터링
- 상세 로깅 기능 추가

## 프로젝트 구조

```
Server2-rag/
├── .dockerignore             # 도커 빌드 시 제외할 파일 설정
├── .env                      # 환경 변수 설정 파일 (API 키 등)
├── .env.example              # 환경 변수 설정 예시 파일
├── Dockerfile                # 도커 이미지 빌드 파일
├── README.md                 # 프로젝트 설명서
├── docker-compose.yml        # 도커 컴포즈 설정 파일
├── requirements.txt          # 파이썬 패키지 의존성 파일
├── data/                     # 임시 데이터 저장 디렉토리
└── src/                      # 소스 코드 디렉토리
    ├── api.py                # FastAPI 웹 서버 및 엔드포인트
    ├── chunker.py            # 텍스트 청킹 기능
    ├── config.py             # 애플리케이션 설정 관리
    ├── evaluator.py          # 청크 요약 및 평가 기능
    ├── graph.py              # 메인 RAG 파이프라인 흐름 관리
    ├── llm.py                # LLM(Google Gemini) 호출 기능
    ├── main.py               # CLI 진입점
    ├── schemas.py            # 데이터 스키마 정의
    └── vector_search.py      # 벡터 검색 API 연동
```

## 파일 세부 기능

### 핵심 파일

#### `src/graph.py`
- 전체 RAG 파이프라인 실행 로직 구현
- 아래 순서로 파이프라인 단계 실행:
  1. 텍스트 청킹 (chunker.py)
  2. 요약 및 검색 쿼리 생성 (evaluator.py)
  3. 검색 필요성 판단 (evaluator.py)
  4. 벡터 검색 API 호출 (외부 API)
  5. 문서 적합성 평가 (evaluator.py)
- 각 청크마다 처리 과정 및 결과를 JSON 형태로 반환

#### `src/chunker.py`
- 입력 텍스트를 의미 단위(청크)로 분할
- NLTK를 사용한 문장 분할 후 RecursiveCharacterTextSplitter로 재그룹
- 설정된 청크 크기(CHUNK_SIZE)와 중복(CHUNK_OVERLAP) 기준 적용

#### `src/llm.py`
- Google Gemini API를 활용한 LLM 호출 기능
- 모델별 최적화된 프롬프트 템플릿 관리
- 재시도 로직(backoff) 구현으로 API 호출 안정성 확보
- 주요 기능:
  - 요약 생성: Gemini 1.5 Flash 사용
  - 판단 및 적합성 평가: Gemini 2.5 Pro Preview 사용
  - 피드백 기반 쿼리 개선
  - 문서 관련성 평가 강화

#### `src/evaluator.py`
- 청크별 요약 및 검색 쿼리 생성 기능
- 검색 필요성 판단 기능(yes/no)
- 문서 적합성 평가 및 피드백 생성
- 재시도 로직 통합

#### `src/config.py`
- Pydantic 기반 애플리케이션 설정 관리
- 환경 변수에서 설정 로드(.env 파일)
- 주요 설정:
  - API 키 및 URL
  - 청크 크기 및 중복 설정
  - 벡터 검색 관련 설정(TOP_K 등)
  - 타임아웃 및 재시도 설정
  - 최대 재시도 횟수
  - 로깅 레벨 설정

#### `src/api.py`
- FastAPI 기반 REST API 구현
- POST /process 엔드포인트 제공
  - 입력: {"text": "회의 텍스트"}
  - 출력: 청크별 RAG 처리 결과

#### `src/main.py`
- CLI 진입점
- STDIN에서 텍스트 입력 받아 처리
- 결과를 JSON 형식으로 STDOUT에 출력

#### `src/vector_search.py`
- 외부 벡터 검색 API 호출 기능
- 검색 결과 및 처리 시간 반환

### 설정 파일

#### `Dockerfile`
- Python 3.11 기반 컨테이너 이미지 설정
- 필요한 패키지 설치 및 NLTK 리소스 다운로드
- FastAPI 서버로 기동(포트 8125)

#### `docker-compose.yml`
- 도커 컴포즈 서비스 설정
- 환경 변수 및 볼륨 설정
- 호스트와 컨테이너 간 포트 매핑(8125:8125)

#### `.env` / `.env.example`
- 필수 환경 변수 설정
  - GEMINI_API_KEY: Google Gemini API 키
  - VECTOR_API_URL: 벡터 검색 API URL
  - 기타 설정 파라미터

## 실행 방법

### 환경 설정

1. `.env` 파일 설정
```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키 및 설정 입력
```

### 도커 컴포즈로 실행

```bash
# 도커 이미지 빌드 및 컨테이너 실행
docker-compose up --build -d

# 로그 확인
docker-compose logs -f
```

### API 호출 예시

```bash
# POST 요청 예시
curl -X POST "http://localhost:8125/process" \
     -H "Content-Type: application/json" \
     -d '{"text": "회의 텍스트 내용을 여기에 입력"}'
```

### CLI 방식으로 실행 (도커 내부)

```bash
# 도커 컨테이너에 접속
docker exec -it agentic_rag bash

# 텍스트 파일을 입력으로 파이프라인 실행
cat input.txt | python -m src.main > output.json
```

## 주요 기능

1. **텍스트 분할**: 문장 단위로 분할 후 의미 단위로 재그룹
2. **요약 및 쿼리 생성**: 각 청크를 요약하고 검색 쿼리 생성
3. **검색 필요성 판단**: LLM이 외부 문서 검색 필요 여부 결정
4. **벡터 검색**: 필요시 외부 벡터 검색 API 활용
5. **문서 적합성 평가**: 검색된 문서와 원 청크의 관련성 평가
6. **JSON 결과 반환**: 모든 과정과 사고 과정을 포함한, 구조화된 결과 제공

## 요구 사항

- Docker 및 Docker Compose
- Python 3.9 이상 (로컬 실행 시)
- 인터넷 연결 (Google Gemini API 및 벡터 검색 API 호출용)
- Google Gemini API 키
- 벡터 검색 API 접근 권한
- 최소 4GB RAM (권장 8GB 이상)
- 2GB 이상의 디스크 여유 공간
