# kch_Final_prj

주가 데이터 + 뉴스 데이터를 결합해 향후 감성분석 모델(KoBERT 파인튜닝 예정)을 학습시키기 위한 데이터 파이프라인 프로젝트.

## 현재 상태 요약
- **완료**: 주가 수집, 뉴스 수집(2가지 방식), 주가 기반 라벨 생성
- **미완료**: 실제 감성분석 모델(학습/추론), 다종목 자동화, market_indicators 데이터 적재
- 즉 지금은 "학습 데이터셋(뉴스 텍스트 X + 라벨 y)을 만드는 준비 단계"이고, 이 데이터로 모델을 학습시키는 코드는 아직 없음

## 프로젝트 구조
```
kch_Final_prj/
├── config.py              # .env 로드, DB_URL / NAVER API 키 관리
├── db_manager.py          # get_db_connection() - 중앙 DB 커넥션 함수
├── docker-compose.yml     # PostgreSQL 16 (stock_db 컨테이너, stockflow_db)
├── requirements.txt
├── Database/
│   ├── create_tables.sql  # daily_stock_prices, daily_news, market_indicators
│   └── init_table.sql     # 전체 테이블 DROP (초기화용, 주의해서 실행)
├── 주가데이터/
│   └── FinanceData_load.py    # FinanceDataReader로 OHLCV 증분 수집
├── 뉴스데이터/
│   ├── NaverNews.py           # 네이버 뉴스 API, 당일 뉴스 수집 (2종목만)
│   ├── NaverNewsCrawl.py      # 네이버 뉴스 크롤링, 변동성 큰 날짜 과거 뉴스 수집
│   └── sentiment_score_label.py  # 주가 등락률 기반 라벨(-2~2) 생성 (텍스트 분석 아님)
└── 쓰래기통/
    └── Bigkinds.py         # 레거시. BigKinds CSV 일괄 적재, 현재 미사용
```

## DB 스키마 핵심
- `daily_stock_prices (ticker, date)` PK — OHLCV + change_rate
- `daily_news (id)` — ticker, date, title, summary, sentiment_score(nullable)
- `market_indicators (indicator_code, date)` PK — **테이블만 정의됨, 적재 스크립트 없음**

## 각 스크립트 상세 동작

**FinanceData_load.py**
1. DB에서 종목별 `MAX(date)` 조회 → 없으면 2020-01-01부터, 있으면 다음날부터 수집
2. 오늘 날짜까지 이미 있으면 스킵
3. FinanceDataReader로 수집 → upsert (`ON CONFLICT DO NOTHING`)
4. `__main__`에 `'005930'`(삼성전자) 하드코딩 — 여러 종목 순회 로직 없음 (TODO)

**NaverNews.py**
- 네이버 뉴스 검색 API로 `STOCK_TARGETS` 종목(현재 삼성전자, SK하이닉스 2개)의 **당일 뉴스**만 수집
- 제목/요약 키워드 스코어링으로 관련성 필터링(`is_relevant_news`)

**NaverNewsCrawl.py**
- 네이버 뉴스 검색 웹페이지 직접 크롤링 (API 아님)
- `daily_stock_prices`에서 등락률 ±3% 이상인 "변동성 큰 날짜" 조회 → 해당 D-1, D 이틀치 뉴스 수집
- 뉴스는 크롤링 시점이 아닌 **변동성 발생일 기준**으로 저장 (주가 매칭 목적)
- 요청 간 랜덤 딜레이(0.7~2초)로 차단 회피

**sentiment_score_label.py**
- ⚠️ 이름과 달리 NLP 감성분석이 아님. 뉴스 텍스트를 읽지 않음
- `daily_news` + `daily_stock_prices` JOIN → 뉴스 발행일의 등락률로 -2~2 규칙 기반 라벨링
- 향후 지도학습용 정답(y) 생성 단계

## 알려진 이슈 / 정리 필요 항목
- `requirements.txt`에 `psycopg`와 `psycopg-binary` 중복 설치되어 있음 — 정리 검토 필요
- `쓰래기통/` 폴더명 오타(의도된 것으로 보임, "안 쓰는 코드 보관용"). `Bigkinds.py`는 레거시, 메인 파이프라인 아님
- `market_indicators` 테이블에 데이터를 넣는 스크립트가 없음
- `docker-compose.yml`의 DB 비밀번호가 하드코딩되어 있었음 → `.env` 참조로 전환 필요 (진행 예정)

## 계획된 다음 단계 (아직 미구현)
1. 다종목 자동화 — 현재 하드코딩된 단일 종목(`005930`) 순회 로직으로 확장
2. KoBERT 등 사전학습 한국어 모델 파인튜닝 — `requirements.txt`에 torch/transformers는 있지만 실제 모델 정의·학습·추론 코드는 아직 없음
3. 뉴스(X) + 라벨(y) 결합한 학습 스크립트 작성
4. market_indicators 데이터 적재 파이프라인
5. 스케줄러/크론으로 수집 자동화 (GPU 필요 작업과 분리해서 컨테이너화 고려 중)

## 환경
- Python venv: `.venv` (Windows 호스트, PyCharm)
- GPU: RTX 4070, torch 2.5.1+cu121 (CUDA 툴킷 시스템 설치 아님, pip 번들형 — 다른 컴퓨터도 드라이버만 맞으면 재현 가능)
- DB: Docker로 Postgres 16만 구동(`docker-compose.yml`), 앱 코드는 컨테이너 밖 Windows 호스트에서 실행

## 규칙
- DB 접속 정보, API 키(NAVER_CLIENT_ID/SECRET 등)는 반드시 `.env`에서만 관리, 코드/compose 파일에 하드코딩 금지
- `docker-compose.yml` 환경변수는 `${VAR}` 형식으로 `.env` 참조
- `requirements.txt`는 `pip freeze`로 최신 상태 유지, 중복/불필요 패키지는 정기적으로 정리
- `init_table.sql` 실행 시 전체 테이블이 DROP되므로 반드시 확인 후 실행
