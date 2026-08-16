# kch_Final_prj

주가 데이터 + 뉴스 데이터를 결합해 감성분석 모델(KoBERT)과 가격 예측 모델(Transformer)을 학습시키는 졸업작품 프로젝트.

## 지금 어디까지 왔는가 (세션 시작 시 먼저 읽을 것)

프로젝트는 **두 개의 독립 트랙**으로 병행 진행 중이다.

| 트랙 | 지시서 | 상태 |
|---|---|---|
| **뉴스/감성** (Task A~F) | `TASK_개정판_데이터_재구축.md` | Task A 완료, 판정 승인됨. **Task B 보류 중** |
| **주가/거시** (Task G) | `TASK_G_market_indicators_적재.md` | G-0~G-3 완료. **G-4 착수 대기** |

**다음 할 일: Task G-4** (`시장지표/feature_loader.py` 작성, 누수 없는 피처 조회 함수)

⚠️ 뉴스/감성 트랙은 명시적 지시 없이 재개하지 않는다. Task A 보고서 말미에 "Task B로 넘어가겠습니다"라는 대기 문구가 있으나, 현재는 주가/거시 트랙을 우선하기로 결정했다.

`market_indicators`에 총 9개 지표 적재 완료: KOSPI/KOSDAQ/USD_KRW(G-2, FDR, 14,072행) + 기준금리·국고채3년·국고채10년·CPI·M2·선행지수순환변동치(G-3, ECOS, 24,883행). 반도체 수출금액지수는 종목 특화 지표라 공통 테이블 설계 원칙과 맞지 않아 제외.

---

## 1차 학습 실패와 원인 규명 (완료된 진단)

### 성능 결과

| 설정 | test macro F1 | test accuracy |
|---|---|---|
| 다수결 baseline | 0.0965 | **0.3181** |
| TF-IDF word 1-2gram | 0.1313 | 0.1637 |
| TF-IDF char_wb 2-4gram | **0.1498** | 0.1731 |
| KoBERT 1차 학습 | 0.1473 | 0.1500 |

9200만 파라미터 사전학습 모델과 선형 bag-of-words가 동점이다. **하이퍼파라미터 문제가 아니라 데이터 문제로 확정.**

> ⚠️ **수치 정정**: 이전 문서의 "다수결 baseline 20.9%"는 오기였다. 20.9%는 `score=0`의 비율이며 최빈이 아니다. **test 최빈 클래스는 `score=-2`(31.8%)** 이고 그때의 macro F1은 0.0965다.
> **앞으로 모든 성능 비교의 baseline은 해당 split의 실제 최빈 클래스 기준으로 계산한다.**

### 확정된 근본 원인

**1) 실질 표본 수가 244개다 (최대 병목)**
- 뉴스 10,618행 / 고유 `(ticker, date)` 조합 **244개**. 조합당 기사 중앙값 49건이 같은 라벨 공유
- test는 사실상 약 37일치. 5-class 분류에 244 샘플
- 전량 삼성전자(005930) 단일 종목. SK하이닉스는 0행

**2) `score=0`이 "중립"이 아니라 "2~3% 움직인 날"이었다 (라벨 정의 결함)**
- 수집 하한(등락률 2%)과 중립 상한(3%)이 겹쳐, 라벨×변동성 교차표가 완벽한 대각선
- train의 50.4%를 차지한 "중립" 클래스가 실제로는 "상당히 크게 움직인 날"
- **이 라벨 체계로는 중립의 의미 자체가 성립하지 않는다. 기존 `sentiment_score`는 전면 폐기.**

**3) 동일 텍스트에 상충하는 라벨 12.3%**
- 완전 중복 제목 959개 그룹 / 초과 행 1,148건
- 그중 618개 그룹(1,284행) + 1개 그룹(23행)이 **서로 다른 라벨**을 받음 = 1,307행, 전체의 12.3%
- 원인 2가지:
  - 같은 날 중복 457건 → `NaverNewsCrawl.py:121-126`의 summary 3단 fallback 파서가 같은 기사에서 다른 텍스트를 추출 (통신사 전재 아님)
  - 여러 날 중복 691건 → `build_search_plan`이 D-1과 D를 서로 다른 target_date에 배정. 네이버 검색이 두 질의 모두에서 같은 기사를 반환
- **어떤 모델도 이 구간은 학습 불가. 오차 하한을 직접 규정한다.**

**4) 레짐 이동 + 선택 편향**

| split | 전 거래일 평균 \|cr\| | 뉴스 수집일 평균 \|cr\| | 수집일/거래일 |
|---|---|---|---|
| train (2023-09-01~2026-02-09) | 1.538% | 3.395% | 162/591 (27%) |
| val (~2026-05-14) | 3.694% | 5.158% | 38/62 (61%) |
| test (~2026-07-31) | **5.479%** | 6.679% | 44/54 (**81%**) |

라벨 분포도 완전히 이동한다.

| split | -2 | -1 | 0 | +1 | +2 |
|---|---|---|---|---|---|
| train | 3.9% | 13.5% | **50.4%** | 24.1% | 8.2% |
| val | 15.1% | 6.3% | 32.4% | 20.2% | 26.0% |
| test | **31.8%** | 5.3% | 20.9% | 17.5% | 24.5% |

고정 절대 임계값(수집 2%, 라벨 ±3%/±5%)이 변동성 레짐 변화를 흡수하지 못했다. 게다가 **수집 여부가 Y에 직접 조건화된 선택 편향** 구조이고 그 선택률이 27%→81%로 변한다.

**5) 크롤링 아티팩트**
- 제목 100%(10,618/10,618)에 `새 창 열림` UI 문자열 포함. 종목명과 붙어 토큰 파괴
- 로지스틱 회귀 상위 계수에 `공개새`, `운영새`, `01 네이버뉴스새` 등 UI 텍스트가 피처로 학습됨
- 학습된 특징 상위가 날짜 파편(02, 06, 09, 12), 인명, 제품명 → 모델이 배운 것은 감성이 아니라 **"어느 시기 기사인가"**

### 기각된 가설

- **"두 수집 소스 혼재"** → 기각. 제목 UI 문자열 100% 일치로 단일 소스 확정. `NaverNews.py`(API)는 **DB 기여 0행**
- 실제로는 **같은 크롤러의 두 번 실행** 흔적: 1회차 threshold=0.03(id 1~6055, 6,027행), 2회차 threshold=0.02(id 6056~, 4,591행). `get_significant_volatility_dates` 기본값 0.03 vs `NaverNewsCrawl.py:151`의 0.02
- **"등락률 데이터 오염"** → 기각. 외부 검증 결과 실제 시장 상황과 일치 (아래 참조)

---

## 2026년 시장 레짐 (외부 검증 완료, 데이터는 정상)

test 구간의 이상한 변동성은 데이터 오류가 아니라 실제 사건이다.

- 코스피: 6월 30일 8,476.48 → 7월 31일 6,595.45, **월간 -22.19%**. 7월 28일 시점 월중 -28.9%로 2008년 10월(-23.13%), 1997년 10월(-27.25%) 상회
- 일간 변동: 7월 15일 **+6.24%**, 7월 28일 **-10.84%**
- 서킷브레이커 2026년 한 해 **8회** 발동 (그 이전 역대 전체 6회)
- 삼성전자: 상반기 AI 메모리 슈퍼사이클로 저점 대비 4배 이상 상승, 52주 최고 374,500원(6/19) → 7월 3일 309,500원(-17%). 7월 7일 -7%, 7월 29일 -5%(SK하이닉스 -10%)

### 이 사실의 두 가지 함의

**(a) 초과수익률 라벨링이 필수로 확정됐다**

7월 28일 코스피 -10.84%인 날 삼성전자가 -5%였다면 기존 라벨은 `score=-2`(극단적 악재)지만, 실제로는 **시장 대비 +5.8%p 아웃퍼폼**이다. 라벨이 정반대로 붙는다. 기존 라벨링이 시장 베타를 정답으로 삼고 있었다는 직접 증거.

**(b) test 구간은 뉴스로 예측 가능한 구간이 아니다 (가장 중요)**

폭락의 주된 동인은 **삼성전자·SK하이닉스 단일종목 레버리지 ETF의 강제 청산**으로 분석된다. 레버리지 상품이 자금을 흡수해 본주 가격 기반이 약화된 상태에서 청산 압력과 외국인 매도가 겹쳐 기계적 매도가 발생했다.

**기계적 청산은 뉴스 감성으로 예측할 수 없다.** 정보가 아니라 수급이 가격을 결정한 구간이다. test macro F1 0.147은 모델 결함이라기보다 해당 구간의 가격 형성 메커니즘이 뉴스와 무관했던 결과일 가능성이 높다. → 실패가 아니라 **분석 대상**으로 다룬다 (아래 설계 결정 참조).

---

## 확정된 설계 결정 (기존 지시서에 우선)

Task A 결과와 레짐 검증을 반영해 확정된 사항. `TASK_개정판_데이터_재구축.md`의 해당 부분을 대체한다.

### D-1. 기존 라벨 체계 전면 폐기
`sentiment_score`(-2~2, 고정 절대 임계값)는 비교 기록으로만 보존하고 재사용하지 않는다. Task F 비교표의 "조합 1(기존 라벨)"은 baseline이 아니라 **실패 사례 기록**으로 취급.

### D-2. 라벨 임계값 — 분위수가 아니라 롤링 표준화
전체 기간 분위수로 임계값을 정하면 test 구간 분포 정보가 라벨 정의에 유입되어 **미래 정보 누수**가 발생한다. 채택 방식:

```
excess_return_t = 종목 change_rate_t − KOSPI change_rate_t
sigma_t         = 직전 N거래일(기본 60일) excess_return 표준편차   # t 시점 미포함
z_t             = excess_return_t / sigma_t
label           = z_t에 고정 임계값 적용
```

- `sigma_t`는 t-1까지의 데이터만 사용. 초기 N일은 라벨 NULL
- 임계값 후보: 3-class는 z ±0.5, 5-class는 z ±0.5/±1.5 — **최종 결정은 z 분포 출력 후 사람이 한다**
- 윈도우 N은 20/60/120을 비교해 클래스 분포 안정성 보고

### D-3. 기간 2분할 — 정상 레짐 / 스트레스 구간

| 구간 | 기간(잠정) | 용도 |
|---|---|---|
| 정상 레짐 | 2023-09 ~ 2026-03 | **주 실험**. train/val/test 분할, 성능 측정 |
| 스트레스 | 2026-04 ~ 2026-07 | **별도 케이스 스터디**. 주 실험 성능에 미포함 |

- 경계일은 월별 평균 |등락률| 추이를 출력한 뒤 사람이 확정
- **수집 자체는 전 기간 진행.** 구간 분리는 학습 단계 필터로 처리 (스트레스 구간 데이터는 케이스 스터디에 필요)
- 정상 레짐만 쓰면 표본이 줄어들므로 (a) 시작점 앞당기기 (b) 종목 확장으로 보완. **Task B 착수 전 네이버 금융 종목뉴스의 과거 조회 가능 기간 확인 필수** — 이 값이 (a)의 실현 가능성을 결정

### D-4. 케이스 스터디 분석 항목
주 실험 모델을 스트레스 구간에 적용해: 성능 저하 폭, 무너지는 클래스, 예측 실패일과 서킷브레이커·사이드카 발동일 중첩 여부를 분석.

결론 방향: "뉴스 감성 예측은 정보가 가격을 주도하는 국면에서 작동하며, 수급·기계적 청산이 주도하는 국면에서는 원리적으로 작동하지 않는다." 선행연구(Antweiler & Frank 2004; Tetlock 2007~2011)의 경계 조건을 실증하는 사례가 된다.

### D-5. 종목 순회 병목 2곳 동시 수정
`FinanceData_load.py`의 TICKERS 순회는 정상. 끊긴 곳은 하류 2곳이며 서로 독립적이다.
1. `NaverNewsCrawl.py:150-152` — `stock_code = "005930"` 하드코딩. 종목 순회 자체가 없음
2. `sentiment_score_label.py:62` — `update_news_labels("005930")` 하드코딩. 1번을 고쳐도 라벨은 안 붙음

두 스크립트 모두 **종목별 건수 요약 출력 필수**, 0건 종목은 경고. 예외를 `print`만 하고 삼키는 패턴(`NaverNews.py:100-101`)을 답습하지 않는다.

### D-6. `NaverNews.py` 폐기
DB 기여 0행, 당일 뉴스만 수집해 스케줄러 없이 동작 불가, 예외 묵살. 신규 금융 크롤러가 과거·당일을 모두 커버하므로 존치 이유 없음. `쓰래기통/`으로 이동(삭제 아님). `config.py`의 NAVER 키 코드는 남긴다.

### D-7. 중복 문제의 구조적 해소
- 같은 날 중복 → Task B: 금융 뉴스 페이지는 구조 고정이라 fallback 불필요 + `article_url` UNIQUE
- 여러 날 중복 → Task D: 기사가 `published_at` 하나만 가지므로 `target_date`도 하나만 배정. **구조적으로 재발 불가**
- "날짜로 검색해 그 날짜에 붙이는" 방식에서 "실제 발행 시각으로 귀속시키는" 방식으로 바뀌는 것이 크롤러 교체의 핵심 이득
- 보일러플레이트 정기 코너(`[E Works]삼성전자·LG전자` 22일 23회)는 Task C 후 제목 패턴 필터로 제외. 필터는 설정 상수로 분리

### D-8. 시계열 분할 전략
주 실험은 정상 레짐 구간 내 70/15/15 시계열 분할. **각 split의 레짐 특성(전 거래일 평균 |cr|, 평균 |초과수익률|, KOSPI 평균 |cr|)을 반드시 함께 보고.** Task F 시점에 walk-forward 검증 도입 여부 재논의.

### D-9. 축소안 (시간 부족 시, 사람이 명시적 결정)
5-class → 3-class 단순화. 롤링 z-score ±0.5 기준. **Task F에서 5-class와 병행 비교 권장**

---

## 프로젝트 구조
```
kch_Final_prj/
├── config.py              # .env 로드, DB_URL / NAVER API 키 관리
├── db_manager.py          # get_db_connection() - 중앙 DB 커넥션 함수
├── docker-compose.yml     # PostgreSQL 16 (stock_db 컨테이너, stockflow_db)
├── requirements.txt
├── TASK_개정판_데이터_재구축.md      # 뉴스/감성 트랙 지시서 (Task A~F)
├── TASK_G_market_indicators_적재.md  # 주가/거시 트랙 지시서 (Task G)
├── Database/
│   ├── create_tables.sql  # daily_stock_prices, daily_news, market_indicators
│   ├── init_table.sql     # 전체 테이블 DROP (초기화용, 주의)
│   └── migrations/
│       └── 002_market_indicators.sql  # ✅ 적용 완료
│                                      # (001은 Task B의 news 스키마용으로 예약, 미작성)
├── 주가데이터/
│   └── FinanceData_load.py    # FinanceDataReader로 OHLCV 증분 수집
├── 시장지표/
│   ├── market_index_load.py   # G-2: KOSPI/KOSDAQ/USD_KRW 적재 ✅ 완료
│   ├── ecos_load.py           # G-3: ECOS 거시지표 6종 적재 ✅ 완료
│   ├── db_utils.py            # G-2/G-3 공통 upsert 로직 (indicator_meta/market_indicators)
│   └── feature_loader.py      # G-4: 누수 없는 피처 조회 (신규 생성 예정)
├── 뉴스데이터/
│   ├── NaverNews.py           # ⚠️ DB 기여 0행, 폐기 예정 (쓰래기통 이동)
│   ├── NaverNewsCrawl.py      # ⚠️ Task B에서 NaverFinanceNews.py로 대체 예정
│   ├── NaverFinanceNews.py    # (Task B에서 신규 생성 예정)
│   └── sentiment_score_label.py  # 주가 등락률 기반 라벨 생성 (텍스트 분석 아님)
├── 감성분석/
│   ├── kobert_dataset.py  # daily_news(라벨 有) -> KoBERT Dataset/DataLoader
│   ├── kobert_train.py    # 부분 freeze 파인튜닝 + early stopping
│   ├── baseline_tfidf.py  # TF-IDF + 로지스틱 회귀 baseline + 데이터 진단 지표
│   └── checkpoints/       # best_model.pt, training_history.png (git 미추적)
└── 쓰래기통/
    └── Bigkinds.py         # 레거시. BigKinds CSV 일괄 적재, 미사용
```

---

## DB 스키마

### daily_stock_prices `(ticker, date)` PK
OHLCV + change_rate. 2020-01-02 ~ 2026-07-31, 거래일 1,615일. 005930/000660 각 1,615행

### daily_news `(id)`
ticker, date, title, summary, sentiment_score(nullable)
- **Task B에서 추가 예정**: `published_at`(발행 시각), `target_date`(예측 대상 거래일), `source`, `press`, `article_url`(UNIQUE), `excess_label`

### market_indicators `(indicator_code, date)` PK — ✅ 스키마 확정, 0행

```
indicator_code varchar(32) NOT NULL   -- FK -> indicator_meta
date           date        NOT NULL   -- 기준일(reference date)
value          numeric
unit           varchar(20)            -- ⚠️ DEPRECATED, indicator_meta.unit 사용
published_date date        NOT NULL   -- 실제 공표일
```

### indicator_meta `(indicator_code)` PK — ✅ 신규 생성, 0행

```
indicator_code varchar(32) PK
name           varchar(128) NOT NULL
source         varchar(32)  NOT NULL   -- CHECK IN ('FDR','ECOS')
frequency      varchar(8)   NOT NULL   -- CHECK IN ('D','W','M','Q')
unit           varchar(32)
ecos_stat_code varchar(32)
ecos_item_code varchar(32)
note           text
```

인덱스: PK 2개 + `idx_indicators_date` + `idx_indicators_code_published (indicator_code, published_date DESC)`
FK: `fk_market_indicators_meta` ON UPDATE CASCADE ON DELETE RESTRICT

### market_indicators 사용 계약 (반드시 준수)

1. **적재 순서 강제**: `indicator_meta` upsert → `market_indicators` insert. 메타 없이 넣으면 FK 위반
2. **삭제 순서 강제**: `market_indicators` 삭제 → `indicator_meta` 삭제. ON DELETE RESTRICT
   (CASCADE로 두지 않은 이유: 메타 한 줄 삭제로 지표 데이터가 조용히 사라지는 사고 방지)
   코드 변경은 ON UPDATE CASCADE라 메타만 UPDATE하면 자식이 따라온다
3. **`published_date`는 코드에서 명시 대입.** DEFAULT 없음 — 누락 시 INSERT 즉시 실패
   - 일별 지표: `published_date = date`
   - 월별 지표: 실제 공표일. 확인 어려우면 보수적으로 기준월 다음달 말일 + note에 기록
4. **코드 명명 규칙**: `ECOS_<통계표코드>_<항목코드>` / FDR은 심볼 그대로(`KS11`)
   가장 긴 형태 `ECOS_722Y001_0101000`이 20자로 varchar(32)에 여유 있음
5. **개정치는 최신값만 유지** (이력 미보존). UPSERT 패턴:
   ```sql
   ON CONFLICT (indicator_code, date) DO UPDATE SET value = EXCLUDED.value
     WHERE market_indicators.value IS DISTINCT FROM EXCLUDED.value
   RETURNING (xmax <> 0) AS was_update
   ```
   `xmax <> 0`으로 신규/개정을 구분해 **개정 건수를 로그 출력**할 것
6. **파생값 저장 금지**: 등락률은 `LAG(value)`로 계산. 원본과 불일치 여지 제거

### G-1 제약조건 동작 검증 완료 (전 항목 통과)
메타 없는 코드 INSERT 거부 / `published_date` 생략·NULL 거부 / `source='NAVER'` 거부 / `frequency='Y'` 거부, `'W'` 허용 / 자식 있는 상태 메타 DELETE 거부 / 올바른 순서 DELETE 성공. 테스트는 savepoint 후 롤백, 양 테이블 0행.

---

## 각 스크립트 상세 동작

**FinanceData_load.py**
1. 종목별 `MAX(date)` 조회 → 없으면 2020-01-01부터, 있으면 다음날부터
2. 오늘까지 이미 있으면 스킵 / 3. FDR 수집 → `ON CONFLICT DO NOTHING`
4. `TICKERS` 리스트를 `__main__`에서 순회. try/except로 종목 단위 격리, 마지막에 성공/스킵/실패 요약 출력. **이 순회는 정상 동작 확인됨**

**NaverNews.py** — ⚠️ 폐기 예정
- 네이버 뉴스 검색 API로 `STOCK_TARGETS` 종목의 당일 뉴스만 수집. 키워드 스코어링 필터링
- `STOCK_TARGETS`는 순회하지만 **DB 기여 0행**. 당일만 수집하는 구조라 스케줄러 없이 사실상 동작 안 함
- `NaverNews.py:100-101`에서 `except Exception`으로 에러를 print만 하고 삼킴 → 조용히 0건으로 끝나는 전형적 패턴

**NaverNewsCrawl.py** — ⚠️ Task B에서 대체 예정
- 네이버 뉴스 검색 웹페이지 크롤링. 등락률 임계값 초과일의 D-1, D 이틀치 수집
- `get_significant_volatility_dates` 기본값 0.03, `:151`의 `__main__`은 0.02 전달 → 두 값으로 각각 한 번씩 실행된 흔적이 남음
- 뉴스를 변동성 발생일 기준으로 저장. 요청 간 랜덤 딜레이 0.7~2초
- 알려진 결함: 발행 시각 미수집 / 제목에 UI 텍스트 100% 혼입 / 하루 50건 상한 / `:121-126` summary 3단 fallback 파서 불안정 / `:150-152` 종목 하드코딩 / D-1·D 날짜 이중 배정

**sentiment_score_label.py**
- ⚠️ 이름과 달리 NLP 감성분석 아님. 뉴스 텍스트를 읽지 않음
- `daily_news` + `daily_stock_prices` JOIN → 발행일 등락률로 -2~2 규칙 기반 라벨링
- `:62` `update_news_labels("005930")` 하드코딩 → 종목 순회로 수정 필요

**kobert_dataset.py**
- `load_labeled_news(ticker=None)`: `sentiment_score IS NOT NULL` 행만 date 오름차순 로드, `LABEL_MAP = {-2:0,-1:1,0:2,1:3,2:4}`
- `split_by_ratio()`: 셔플 없이 날짜 순서대로 70/15/15
- `compute_class_weights()`: 빈도 역수 weight tensor
- `NewsDataset`: title+summary를 sentence-pair 토크나이징
- 토크나이저 `AutoTokenizer.from_pretrained('skt/kobert-base-v1')` — sentencepiece 에러 없이 정상(`monologg/kobert` 불필요). 단 XLNetTokenizer 기반이라 `return_token_type_ids=True` 명시 필요, 세그먼트 id가 0/1/2로 나오는데 `type_vocab_size=2`라 `clamp(max=1)`로 방어
- **Task F 예정 변경**: `label_column` 파라미터 추가해 `sentiment_score`/`excess_label` 선택 가능하게

**baseline_tfidf.py**
- `kobert_dataset.py`의 로더/분할을 재사용해 동일 조건 TF-IDF + 로지스틱 회귀
- word 1-2gram / char_wb 2-4gram 비교, `class_weight='balanced'`
- 함께 출력: split별 라벨 분포, `(ticker,date)` 조합 수·조합당 기사 수, 중복 제목 통계, 라벨별 상위 특징 단어

**kobert_train.py** — ⚠️ **데이터 재구축 완료 전까지 재학습하지 않는다**
- 모델 `BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=5)`
- 부분 freeze: embeddings + 인코더 0~8 고정, 9/10/11 + pooler + classifier만 학습. 92,190,725개 중 21,858,053개(23.7%)
- `AdamW(lr=1e-5)` + `get_linear_schedule_with_warmup`(warmup 10%), class weight `CrossEntropyLoss`
- early stopping: val macro F1 기준 `patience=2`, 최대 10 epoch, batch 16
- best checkpoint 저장, test `classification_report` 출력, 학습 곡선 png 저장
- 1차 학습: 5 epoch early stopping. val macro F1이 3 epoch 이후 하락(0.1792→0.1692→0.1626)하는데 train_loss는 감소 → 76% freeze + lr=1e-5 세팅에서도 과적합

---

## 알려진 이슈 / 정리 필요 항목
- `requirements.txt`에 `psycopg`와 `psycopg-binary` 중복 — 정리 검토
- `docker-compose.yml` DB 비밀번호 하드코딩 → `.env` 참조로 전환 필요
- `market_indicators.unit` DEPRECATED — 적재 시 NULL로 두고 `indicator_meta.unit` 사용
- `migrations/001`이 비어 있음 (Task B의 news 스키마용 예약 번호)
- `쓰래기통/` 폴더명 오타는 의도된 것("안 쓰는 코드 보관용")

---

## 로드맵

### 주가/거시 트랙 (Task G) — 진행 중
| 단계 | 내용 | 상태 |
|---|---|---|
| G-0 | 스키마 확인 | ✅ 완료 |
| G-1 | 스키마 설계 + 마이그레이션 + 제약 검증 | ✅ 완료 |
| G-2 | `시장지표/market_index_load.py`, KOSPI/KOSDAQ/USD_KRW 적재 | ✅ 완료 (14,072행) |
| G-3 | ECOS 지표 코드 조회 → 적재 (기준금리·국고채3·10년·CPI·M2·선행지수) | ✅ 완료 (24,883행) |
| **G-4** | **`feature_loader.py` — 누수 없는 피처 조회 + 검증** | **← 다음** |

G-2/G-3 완료로 KOSPI 등락률과 거시지표 6종이 확보되어 초과수익률 라벨링(Task E)이 열린다.

### 뉴스/감성 트랙 (Task A~F) — 보류
| 단계 | 내용 | 상태 |
|---|---|---|
| A | 원인 확정 진단 | ✅ 완료, 판정 승인 |
| B | 수집기 교체(네이버 금융 종목뉴스), 발행 시각 확보 | 보류 |
| C | 전면 재수집. 변동성 필터 제거, 섹터 분산 5~8종목 (목표 고유 조합 2,000+) | 보류 |
| D | 이벤트 윈도우 재정렬 (X=장 시작 전 기사, Y=당일 등락률) | 보류 |
| E | 초과수익률 라벨링 + 롤링 표준화 | 보류 |
| F | 조합 비교 실험 (윈도우 × 라벨) | 보류 |

**Task B 착수 전 확인 필수**: 네이버 금융 종목뉴스 페이지의 과거 조회 가능 기간 (005930 기준 2020/2022/2023년 접근 테스트)

### Transformer 트랙 — 미착수
G-4 완료 후 착수. **감성 피처 없이 주가 + 거시지표만으로 학습한 결과가 ablation baseline이 된다.** 이 baseline 없이는 "뉴스 감성 추가로 개선되었는가"를 증명할 수 없으므로, 감성 트랙보다 먼저 만들어두는 것이 합리적이다.

### 이후
스케줄러/크론 자동화, 추론(서빙) 코드, 컨테이너화(Docker → Kubernetes, 최후순위)

---

## Task C 종목 확장 원칙 (섹터 분산)

현재 실질 분석 대상은 삼성전자 1종목. SK하이닉스(000660)는 확장성 검증용 자리표시자로 등록만 되어 있고 뉴스 0행.

선행연구(정지선·김동성·김종우, 2015, 지능정보연구 21권 4호)에서 **IT 섹터의 뉴스 기반 예측 정확도가 전 섹터 중 최저**였다. IT만으로 구성하면 성능 저하 시 "종목 선정 문제"와 "방법론 문제"가 구분되지 않는다. 섹터를 섞으면 섹터별 성능 비교가 독립적인 분석 축이 된다.

후보: 005930 삼성전자 / 005380 현대차 / 051910 LG화학 / 105560 KB금융 / 207940 삼성바이오로직스 / 035420 NAVER
(000660은 코드 유지, **비활성** — 수집 대상 제외)

---

## 환경
- Python 3.12, venv `.venv` (Windows 호스트, PyCharm + Claude Code CLI)
- GPU: RTX 4070, torch 2.5.1+cu121 (pip 번들형, CUDA 툴킷 시스템 설치 아님)
- DB: Docker Postgres 16만 구동, 앱 코드는 컨테이너 밖 Windows 호스트

---

## 규칙
- DB 접속 정보·API 키는 `.env`에서만 관리. 코드/compose 하드코딩 금지
- `docker-compose.yml` 환경변수는 `${VAR}` 형식으로 `.env` 참조
- `requirements.txt`는 `pip freeze`로 최신 유지
- `init_table.sql` 실행 시 전체 테이블 DROP — 반드시 확인 후 실행
- **스키마 변경은 `ALTER TABLE`로 하고 `Database/migrations/`에 SQL 파일로 남긴다. 기존 테이블을 DROP하지 않는다**
- **기존 컬럼 값을 파괴적으로 덮어쓰지 않는다. 새 라벨·새 시각은 새 컬럼에 추가해 비교 가능하게 둔다**
- 모든 스크립트는 프로젝트 루트에서 `python -m <패키지>.<모듈>` 형태로 실행 가능해야 한다
- **모든 성능 비교 baseline은 해당 split의 실제 최빈 클래스 기준** (현재 test 31.8% / macro F1 0.0965)
- **예외를 `print`만 하고 삼키지 않는다.** 단위별 격리 + 최종 요약에 실패 항목 명시 + 0건 항목 경고
- 각 Task의 정지 지점에서 임의 판단으로 다음 단계에 진입하지 않는다
