# kch_Final_prj

주가 데이터 + 뉴스 데이터를 결합해 향후 감성분석 모델(KoBERT 파인튜닝 예정)을 학습시키기 위한 데이터 파이프라인 프로젝트.

## 현재 상태 요약
- **완료**: 주가 수집(다종목 자동화 포함), 뉴스 수집(2가지 방식), 주가 기반 라벨 생성, KoBERT 학습용 데이터 로더/Dataset 구현, KoBERT 파인튜닝 학습 루프(부분 freeze + early stopping) 구현 및 1차 전체 학습 실행
- **미완료**: 실제 추론(서빙) 코드, 뉴스 수집 종목 확장(현재 2종목 하드코딩), market_indicators 데이터 적재
- ⚠️ **1차 학습 결과 성능 미달 — 다음 세션 최우선 이슈**: 5 epoch에서 early stopping, test macro F1=0.1473, test accuracy=15%. test set 다수 클래스(label=0, score=0) 비율이 20.9%인데 모델 정확도(15%)가 그보다 낮아 단순 다수결 baseline에도 못 미침. val macro F1이 3 epoch 이후 계속 하락(0.1792→0.1692→0.1626)하는데 train_loss는 계속 감소하는 과적합 패턴이 76% freeze + lr=1e-5의 보수적 세팅에서도 나타남 → 모델 용량 문제라기보다 라벨 자체의 신호가 약할 가능성. 가장 유력한 가설: 현재 라벨링(`daily_stock_prices.change_rate` 절대값 기준)이 코스피 동반 상승/하락과 종목 고유 반응을 구분하지 못해서 뉴스 텍스트와 라벨 간 실제 상관관계가 약함 (자세한 다음 단계는 아래 "계획된 다음 단계" 참고)

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
├── 감성분석/
│   ├── kobert_dataset.py  # daily_news(라벨 有) -> KoBERT 입력 Dataset/DataLoader 준비
│   ├── kobert_train.py    # 부분 freeze 파인튜닝 학습 루프 + early stopping + 평가
│   └── checkpoints/       # best_model.pt, training_history.png (학습 실행 시 생성, git 미추적)
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
4. 코드 상단 `TICKERS` 리스트(현재 `005930` 삼성전자, `000660` SK하이닉스)를 `__main__`에서 순회하며 종목별로 `update_stock_data(ticker)` 호출. 종목 하나가 실패해도 나머지는 계속 진행되도록 try/except로 격리하고, 마지막에 종목별 성공/스킵/실패 결과를 요약 출력. 종목 추가는 `TICKERS`에 코드만 추가하면 됨

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

**kobert_dataset.py**
- `load_labeled_news(ticker=None)`: `daily_news`에서 `sentiment_score IS NOT NULL`인 행만 date 오름차순으로 로드하고, `LABEL_MAP = {-2:0, -1:1, 0:2, 1:3, 2:4}`로 5클래스 `label` 컬럼 추가
- `split_by_ratio()`: 셔플 없이 날짜 순서 그대로 train(70%)/val(15%)/test(15%) 분할
- `compute_class_weights()`: 클래스별 빈도 역수로 weight tensor 계산 (`CrossEntropyLoss(weight=...)`용)
- `NewsDataset`: title+summary를 sentence-pair로 토크나이징해 `input_ids/attention_mask/token_type_ids/label` 반환
- 토크나이저는 `AutoTokenizer.from_pretrained('skt/kobert-base-v1')` 사용 — 실제 로드 확인 결과 sentencepiece 에러 없이 정상 동작(`monologg/kobert` 대체 불필요). 단 이 토크나이저가 XLNetTokenizer 기반이라 `return_token_type_ids=True`를 명시해야 token_type_ids가 나오고, 세그먼트 id가 0/1/2로 나오는데 실제 모델(BertConfig)의 `type_vocab_size=2`라 2가 들어오면 임베딩 인덱스 에러가 남 → Dataset에서 `clamp(max=1)`로 방어
- 다른 하위 폴더 스크립트와 마찬가지로 `db_manager` import 때문에 프로젝트 루트가 `PYTHONPATH`에 잡혀 있어야 실행됨

**kobert_train.py**
- `kobert_dataset.py`의 `load_labeled_news/split_by_ratio/compute_class_weights/NewsDataset`을 그대로 재사용
- 모델: `BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=5)`
- **부분 freeze**: `model.bert.embeddings` + 인코더 레이어 0~8(하위 9개) 고정, 나머지 9/10/11번 레이어 + pooler + classifier만 학습(`requires_grad=True`). 학습 시작 시 총/학습가능 파라미터 수 출력(RTX 4070 기준 실측: 총 92,190,725개 중 21,858,053개(23.7%)만 학습)
- optimizer `AdamW(lr=1e-5)`(freeze 안 된 파라미터만 전달) + `get_linear_schedule_with_warmup`(warmup 10%), loss는 `compute_class_weights`로 만든 class weight를 적용한 `CrossEntropyLoss`
- **early stopping**: 매 epoch 종료 시 val loss/macro F1 계산, val macro F1이 개선 안 된 epoch가 `patience=2`번 연속되면 중단. 최대 10 epoch, batch size 16(주석에 VRAM에 따라 조정 가능하다고 명시)
- 개선될 때마다 `감성분석/checkpoints/best_model.pt`에 `state_dict` 저장, 학습 종료 후 best checkpoint로 test set `classification_report` 출력(class index -> 실제 라벨(-2~2) 역매핑도 같이 표시)
- 학습 곡선(train/val loss, val macro F1)을 `감성분석/checkpoints/training_history.png`로 저장
- 실행: `python 감성분석/kobert_train.py` (역시 프로젝트 루트가 `PYTHONPATH`에 있어야 함). 스모크 테스트(소규모 서브셋 2 epoch, GPU shape/메모리 에러 없음) 이후 전체 데이터로 1차 학습을 실행함 — 5 epoch에서 early stopping, test macro F1=0.1473/accuracy=15%로 다수결 baseline보다도 낮게 나옴. 결과 해석과 원인 가설은 위 "현재 상태 요약", 다음 조치는 아래 "계획된 다음 단계" 참고

## 알려진 이슈 / 정리 필요 항목
- `requirements.txt`에 `psycopg`와 `psycopg-binary` 중복 설치되어 있음 — 정리 검토 필요
- `쓰래기통/` 폴더명 오타(의도된 것으로 보임, "안 쓰는 코드 보관용"). `Bigkinds.py`는 레거시, 메인 파이프라인 아님
- `market_indicators` 테이블에 데이터를 넣는 스크립트가 없음
- `docker-compose.yml`의 DB 비밀번호가 하드코딩되어 있었음 → `.env` 참조로 전환 필요 (진행 예정)

## 계획된 다음 단계 (아직 미구현)
1. **(최우선) KoBERT 1차 학습 성능 미달 원인 규명** — 1차 학습 결과 test macro F1=0.1473으로 다수결 baseline(약 20.9%)보다도 낮음(위 "현재 상태 요약" 참고). 합의된 진행 순서:
   a. TF-IDF + 로지스틱 회귀 같은 가벼운 baseline을 동일 데이터로 학습시켜, 문제가 "라벨 신호 자체의 부족"인지 "KoBERT 쪽 하이퍼파라미터/설정 문제"인지부터 분리해서 확인
   b. 라벨링을 초과수익률(종목 등락률 − 코스피 등락률) 기준으로 개선한 뒤 재학습 — 기존 로드맵에 있던 "라벨링 로직 개선"을 지금 시점으로 우선순위 상향
   c. 대안으로 5-class(-2~2) → 3-class(하락/중립/상승) 라벨 단순화도 검토 후보
2. 종목 확장 — 주가 수집(`FinanceData_load.py`)은 `TICKERS` 리스트 순회로 다종목 자동화 완료(현재 2종목). 뉴스 수집(`NaverNews.py`)은 아직 `STOCK_TARGETS` 2개 하드코딩이라 동일한 방식으로 확장 필요
3. market_indicators 데이터 적재 파이프라인
4. 스케줄러/크론으로 수집 자동화 (GPU 필요 작업과 분리해서 컨테이너화 고려 중)

## 환경
- Python venv: `.venv` (Windows 호스트, PyCharm)
- GPU: RTX 4070, torch 2.5.1+cu121 (CUDA 툴킷 시스템 설치 아님, pip 번들형 — 다른 컴퓨터도 드라이버만 맞으면 재현 가능)
- DB: Docker로 Postgres 16만 구동(`docker-compose.yml`), 앱 코드는 컨테이너 밖 Windows 호스트에서 실행

## 규칙
- DB 접속 정보, API 키(NAVER_CLIENT_ID/SECRET 등)는 반드시 `.env`에서만 관리, 코드/compose 파일에 하드코딩 금지
- `docker-compose.yml` 환경변수는 `${VAR}` 형식으로 `.env` 참조
- `requirements.txt`는 `pip freeze`로 최신 상태 유지, 중복/불필요 패키지는 정기적으로 정리
- `init_table.sql` 실행 시 전체 테이블이 DROP되므로 반드시 확인 후 실행
