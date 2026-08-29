# kch_Final_prj

주가 데이터 + 뉴스 데이터를 결합해 감성분석 모델(KoBERT)과 가격 예측 모델(Transformer)을 학습시키는 졸업작품 프로젝트.

## 지금 어디까지 왔는가 (세션 시작 시 먼저 읽을 것)

프로젝트는 **두 개의 독립 트랙**으로 병행 진행 중이다.

| 트랙 | 지시서 | 상태 |
|---|---|---|
| **뉴스/감성** (Task A~F) | `TASK_개정판_데이터_재구축.md`(A~D) + `TASK_EF_라벨링_비교실험.md`(E~F, 2026-08-29 분리) | Task A 완료, 판정 승인됨. **Task B 재개 — 전면 백필 진행 중(77.1%, 2026-08-29 기준)** |
| **주가/거시** (Task G) | `TASK_G_market_indicators_적재.md` | **G-0~G-4 전체 완료** |

**현재 상태(2026-08-29)**: Task G 완료 후 열렸던 두 갈래(Transformer 착수 / 뉴스·감성 Task B 재개) 모두
착수됐다. **Task T-1(Transformer ablation baseline)은 완료돼 "가격+거시지표만으로는 익일 방향성이
무작위 수준과 구분 안 됨"으로 확정**됐다(상세는 "파이프라인 실행 순서 > 트랙 2" 참고). **뉴스/감성
Task B도 재개돼 전면 백필이 진행 중**이다(체크포인트 기준 77.1%, 상세는 로드맵 절 참고).

`market_indicators`에 총 9개 지표, 38,955행(=14,072+24,883) 적재 완료: KOSPI/KOSDAQ/USD_KRW(G-2, FDR, 14,072행) + 기준금리·국고채3년·국고채10년·CPI·M2·선행지수순환변동치(G-3, ECOS, 24,883행). 반도체 수출금액지수는 종목 특화 지표라 공통 테이블 설계 원칙과 맞지 않아 제외. `시장지표/feature_loader.py`(G-4)로 누수 없는 피처 조회 가능 — `get_features(ticker, start_date, end_date)`가 종목 OHLCV + 지표 9개를 `published_date < 거래일` 조건으로 병합해 반환.

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

⚠️ **2026-08-29 정정**: 위 `NaverNewsCrawl.py`/`sentiment_score_label.py` 두 파일 모두
2026-08-23에 삭제됨 — 라인 번호 참조는 죽은 참조다. 다만 그 대체 파일(`뉴스_최초적재.py`/
`뉴스_일일수집.py`)은 애초에 `tickers`를 함수 인자로 받는 구조라 이 병목 자체가 구조적으로
이미 해소된 상태다.

### D-6. `NaverNews.py` 폐기
DB 기여 0행, 당일 뉴스만 수집해 스케줄러 없이 동작 불가, 예외 묵살. 신규 금융 크롤러가 과거·당일을 모두 커버하므로 존치 이유 없음. `쓰래기통/`으로 이동(삭제 아님). `config.py`의 NAVER 키 코드는 남긴다.

⚠️ **2026-08-29 정정**: 실제로는 "쓰래기통으로 이동"이 아니라 2026-08-23 감사 때 다른 7개
폐기 파일과 함께 **완전 삭제**됐다(git 이력에는 남아있어 복구 가능 — 실제로 이번 세션에
`쓰래기통/Bigkinds.py`를 이 방식으로 복구한 사례 있음). `config.py`의 NAVER 키 코드는 계획대로 남아있다.

### D-7. 중복 문제의 구조적 해소
- 같은 날 중복 → Task B: 금융 뉴스 페이지는 구조 고정이라 fallback 불필요 + `article_url` UNIQUE
- 여러 날 중복 → Task D: 기사가 `published_at` 하나만 가지므로 `target_date`도 하나만 배정. **구조적으로 재발 불가**
- "날짜로 검색해 그 날짜에 붙이는" 방식에서 "실제 발행 시각으로 귀속시키는" 방식으로 바뀌는 것이 크롤러 교체의 핵심 이득
- 보일러플레이트 정기 코너(`[E Works]삼성전자·LG전자` 22일 23회)는 Task C 후 제목 패턴 필터로 제외. 필터는 설정 상수로 분리

### D-8. 시계열 분할 전략
주 실험은 정상 레짐 구간 내 70/15/15 시계열 분할. **각 split의 레짐 특성(전 거래일 평균 |cr|, 평균 |초과수익률|, KOSPI 평균 |cr|)을 반드시 함께 보고.** Task F 시점에 walk-forward 검증 도입 여부 재논의.

### D-9. 축소안 (시간 부족 시, 사람이 명시적 결정)
5-class → 3-class 단순화. 롤링 z-score ±0.5 기준. **Task F에서 5-class와 병행 비교 권장**

### D-10. 데이터 소스 정책 — `daily_news`가 정본, BigKinds는 실험 자산 (2026-08-29)
`daily_news`(크롤링, search_backfill)를 학습·운영 **정본**으로 삼는다. `daily_news_bigkinds`
(BigKinds)는 매일 자동 수집이 불가능(유료, 수동 다운로드)해 MLOps 자동화 요건을 충족하지
못하므로 정본이 될 수 없다.

`daily_news_bigkinds`는 (a) 데이터 소스 품질 비교, (b) **과거 데이터를 빅카인즈로 대체
가능한지 검증**하는 실험 자산으로 사용한다 — (b)가 검증되면 향후 종목·기간 확장 시 초기 학습
데이터 확보 비용이 크게 줄어든다. 상세 실험 설계(데이터소스 교차 평가 등)는
`TASK_EF_라벨링_비교실험.md` 참고.

---

## 프로젝트 구조
```
kch_Final_prj/
├── config.py              # .env 로드, DB_URL / NAVER API 키 관리
├── db_manager.py          # get_db_connection() - 중앙 DB 커넥션 함수
├── docker-compose.yml     # PostgreSQL 16 (stock_db 컨테이너, stockflow_db)
├── requirements.txt
├── TASK_개정판_데이터_재구축.md      # 뉴스/감성 트랙 지시서 (Task A~D, 완료 아카이브)
├── TASK_EF_라벨링_비교실험.md        # 뉴스/감성 트랙 지시서 (Task E~F, 2026-08-29 분리)
├── TASK_G_market_indicators_적재.md  # 주가/거시 트랙 지시서 (Task G)
├── Database/              # 2026-08-23 도메인별 폴더로 재구성 (옛 create_tables.sql/init_table.sql/migrations/ 삭제)
│   ├── 뉴스/               # 테이블_생성.sql(daily_news) / 데이터_삭제.sql / 테이블_삭제.sql
│   ├── 주가/               # 테이블_생성.sql(daily_stock_prices) / 데이터_삭제.sql / 테이블_삭제.sql
│   └── 시장지표/            # 테이블_생성.sql(indicator_meta→market_indicators, FK 순서) / 데이터_삭제.sql / 테이블_삭제.sql
├── 주가데이터/
│   └── FinanceData_load.py    # FinanceDataReader로 OHLCV 증분 수집
├── 시장지표/
│   ├── market_index_load.py   # G-2: KOSPI/KOSDAQ/USD_KRW 적재 ✅ 완료
│   ├── ecos_load.py           # G-3: ECOS 거시지표 6종 적재 ✅ 완료
│   ├── db_utils.py            # G-2/G-3 공통 upsert 로직 (indicator_meta/market_indicators)
│   └── feature_loader.py      # G-4: 누수 없는 피처 조회 ✅ 완료
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

2026-08-23부터 `Database/{도메인}/테이블_생성.sql`이 "지금 스키마"의 유일한 정본이다(운영 DB를
`information_schema`/`pg_constraint`/`pg_indexes`로 직접 조회해 실측 검증 완료 — 격리된 임시
스키마에 재생성해 컬럼/제약/인덱스 단위까지 100% 일치 확인). 아래는 그 요약이며, 상세 타입/제약은
해당 SQL 파일을 참고할 것. 옛 `Database/create_tables.sql`/`init_table.sql`/`migrations/`는
삭제됨(이 SQL 파일들로 대체).

**부가 발견**: DB에 `daily_news_backup` 테이블(마이그레이션 이전 구 6컬럼 스키마 스냅샷, 5,278행,
CLAUDE.md/코드 어디에도 문서화 안 됨)이 있었음 — FK/인덱스 없음 확인 후 사람 승인으로 삭제됨
(2026-08-23).

### daily_stock_prices `(ticker, date)` PK — `Database/주가/테이블_생성.sql`
OHLCV + change_rate. 2020-01-02 ~ 2026-07-31, 거래일 1,615일. 005930/000660 각 1,615행

### daily_news `(id)` — `Database/뉴스/테이블_생성.sql`
`ticker, date, title, summary, sentiment_score`(구 5-tier 라벨, 폐기 대상, nullable) +
`published_at, target_date, source, press, article_url`(migrations/001·003이 반영된 상태,
2026-08-23 기준 이미 라이브 — "추가 예정" 아님). `target_date`는 Task D 완료 전이라 현재 전부 NULL.
`UNIQUE(ticker, article_url) WHERE article_url IS NOT NULL`로 종목별 기사 중복 방지.

### daily_news_bigkinds `(id)` — `Database/뉴스_빅카인즈/테이블_생성.sql`
BigKinds(빅카인즈) CSV/Excel 일괄 적재 전용 — `daily_news`와 완전히 분리된 별도 테이블(FK 없음,
서로 영향 없음). `title, summary`(본문 전체), `press, article_url, keywords, category`(통합 분류1),
`bigkinds_id`(뉴스 식별자), `target_date`(Task D 윈도우 재정렬용, 현재 NULL), `source` DEFAULT
'bigkinds'. `UNIQUE(bigkinds_id)` + `UNIQUE(ticker, article_url) WHERE article_url IS NOT NULL`로
중복 방지(2026-08-29, `빅카인즈/빅카인즈_적재.py`).

⚠️ **`뉴스 식별자`를 읽을 때 반드시 `dtype=str`을 강제할 것.** 지정하지 않으면 pandas가 float64로
자동 추론하는데, 실제 값이 26자리(예: `02100501.20260829080426001`)라 float64 유효자릿수(15~17자리)를
넘어 뒷부분이 잘린다. 언론사+날짜 앞자리가 같은 서로 다른 기사가 같은 값으로 뭉개져 UNIQUE 제약이
진짜 다른 기사를 "중복"으로 오판해 조용히 버리는 데이터 유실이 실제로 발생했다(2026-08-29 발견 —
한 파일에서 66% 행이 충돌, 최대 23개 별개 기사가 한 값으로 뭉개진 사례 확인 후 수정). `빅카인즈_적재.py`
에는 이미 반영돼 있음 — 참고할 때 이 점을 잊지 말 것.

2026-08-29 최초 적재 완료: 20,053행(전체 읽음 20,221행, 파일 내부 정상 중복 168건 제외), 날짜 범위
2023-08-23~2026-08-29, 전량 005930 고정. `daily_news`(search_backfill 등)와 어떻게 통합·비교할지는
아직 미결 — Task A~F 어느 단계에도 아직 편입되지 않은 독립 데이터 소스다.

### market_indicators `(indicator_code, date)` PK — `Database/시장지표/테이블_생성.sql`

```
indicator_code varchar(32) NOT NULL   -- FK -> indicator_meta
date           date        NOT NULL   -- 기준일(reference date)
value          numeric
unit           varchar(20)            -- ⚠️ DEPRECATED, indicator_meta.unit 사용
published_date date        NOT NULL   -- 실제 공표일
```

### indicator_meta `(indicator_code)` PK — `Database/시장지표/테이블_생성.sql`

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
4. **코드 명명 규칙**: `ECOS_<통계표코드>_<항목코드>` / FDR은 사람이 읽기 쉬운 이름 사용
   (실제 저장값: `KOSPI`/`KOSDAQ`/`USD_KRW` — ⚠️ 2026-08-29 정정: 이전에 "FDR은 심볼 그대로
   `KS11`"이라 적어뒀으나 실제 DB 조회 결과 raw FDR 심볼이 아니라 이 이름들로 저장돼 있었음)
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

(2026-08-23에 삭제된 `FinanceData_load.py`/`NaverNews.py`/`NaverNewsCrawl.py`/`sentiment_score_label.py`의
상세 동작 설명은 제거됨 — 각 파일 한 줄 요약은 아래 "파이프라인 실행 순서" 섹션의 "삭제 완료" 목록 참고.)

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

## 파이프라인 실행 순서 (전체 .py 파일 감사, 2026-08-23)

프로젝트 내 모든 `.py` 파일(30개, `.venv` 제외)을 대상으로 한 감사 결과. 목적별로 실행 순서와
상태를 정리한다. 상태 분류: **핵심 파이프라인**(현재 설계에 실제로 쓰이는 흐름) /
**실험·진단용**(일회성 분석·비교, 재학습 파이프라인 아님) / **폐기 후보**(대체됨·미사용).

### 인프라 (전 트랙 공통)

| 파일 | 역할 | 상태 |
|---|---|---|
| `config.py` | `.env` 로드, `DB_URL`/`NAVER_*`/`ECOS_API_KEY` 제공 | 핵심 |
| `db_manager.py` | `get_db_connection()` — 사실상 전 파일이 의존 | 핵심 (의존: `config`) |
| `constants.py` | Task T 전용 스트레스 구간 상수(`STRESS_PERIOD_START/END`, 하드코딩, 자동 재탐지 금지 설계) | 핵심 |

### 트랙 1 — 주가/거시 (Task G, 전체 완료 ✅)

2026-08-23 구조 정리로 초기적재/일일수집이 파일 단위로 분리됨(같은 날 재작업: 처음엔 초기적재가
일일수집 함수를 직접 import하는 종속 구조였다가, 뉴스 트랙의 `뉴스_공통.py` 패턴에 맞춰
`주가_공통.py`/`지표_공통.py`를 두 파일이 대등하게 참조하는 구조로 다시 정리함 — 초기적재와
일일수집은 서로를 import하지 않는다). 날짜 상수는 `constants.py`(`STOCK_INITIAL_LOAD_START`)로
중앙화 — 시장지표 로더는 "고정 시작일" 개념 자체가 없어(전체 기간 조회가 기본 동작) 상수화
대상에서 제외했다(사람 확인 완료).

| 순서 | 파일 | 역할 | 의존성 |
|---|---|---|---|
| 1 | `주가데이터/주가_공통.py` | `TICKERS`, `update_stock_data()`(핵심 fetch+upsert) 등 공용 로직 | `db_manager` |
| 1a | `주가데이터/주가_초기적재.py` | 데이터 없는 종목만 `constants.STOCK_INITIAL_LOAD_START`부터 전체 적재 | `constants`, `주가데이터.주가_공통` |
| 1b | `주가데이터/주가_일일수집.py` | 종목별 DB 최신일+1부터 오늘까지 증분 수집 | `주가데이터.주가_공통` |
| 2 | `시장지표/db_utils.py` | upsert 공용 로직 | 없음 (다른 모듈이 가져다 씀) |
| 3 | `시장지표/지표_공통.py` | FDR(`load_fdr_indicators`)·ECOS(`load_ecos_indicators`) 함수. 두 섹션 독립 예외처리 — 한쪽 실패해도 다른 쪽 계속 진행 | `config`, `db_manager`, `시장지표.db_utils` |
| 3a | `시장지표/지표_초기적재.py` | FDR+ECOS 중 아직 한 번도 적재 안 된 지표만 초기적재(`only_uninitialized=True`) | `db_manager`, `시장지표.지표_공통` |
| 3b | `시장지표/지표_일일수집.py` | FDR+ECOS 증분 적재(`only_uninitialized=False`) | `db_manager`, `시장지표.지표_공통` |
| 4 | `시장지표/feature_loader.py` | 누수 없는 피처 조회(`get_features`) — 트랙 2 전체가 이 함수 하나에 의존 | `db_manager` |

**옛 파일 삭제 완료(2026-08-23)**: `주가데이터/FinanceData_load.py`(종목별 OHLCV를 FDR로 증분
수집하던 단일 파일 — 정상 동작이었음, → 주가_초기적재/일일수집으로 분리),
`시장지표/market_index_load.py`(→ 지표_초기적재/일일수집의 FDR 섹션으로 분리),
`시장지표/ecos_load.py`(→ 지표_초기적재/일일수집의 ECOS 섹션으로 분리). 삭제 전 grep으로 실제 import 의존
없음을 확인했다(서로를 참조하던 것 외 다른 파일에서의 import 없음, 주석 언급만 존재).

### 트랙 2 — Transformer 가격예측 (Task T-1, 완료 ✅ — "무작위 수준" 확정, 트랙 1 재개 근거로 연결됨)

`가격예측/` 13개 파일 전부 실제로 쓰였음(폐기 후보 없음). `TASK_T_transformer_baseline.md`의
진단 순서와 정확히 대응한다.

**핵심 파이프라인:**

| 순서 | 파일 | 역할 | 의존성 |
|---|---|---|---|
| 1 | `가격예측/dataset_builder.py` | 레이블(익일수익률)+기본피처 계산. `build_base_dataset`(v1)/`build_base_dataset_v2`(레벨→비율 변환) | `시장지표.feature_loader` |
| 2 | `가격예측/momentum_feature.py` | lag 적용 초과수익률(z-score) 모멘텀 피처 | `db_manager`, `시장지표.feature_loader` |
| 3 | `가격예측/split_dataset.py` | base+momentum merge, 정상레짐/스트레스 분리, 70/15/15 분할(v1/v2 겸용) | `constants`, `가격예측.dataset_builder`, `가격예측.momentum_feature` |
| 4 | `가격예측/model.py` | `TransformerRegressor` 정의 | 없음 |
| 4 | `가격예측/sequence_dataset.py` | lookback 시퀀스 변환, `FeatureScaler` (4와 병렬) | 없음 |
| 5 | `가격예측/train.py` | v1 베이스라인 학습 + **test 1회 평가**(체크포인트4). ⚠️ `train_common.py`를 안 쓰고 자체 헬퍼(`directional_accuracy` 등) 보유 | `constants`, `가격예측.model`, `가격예측.sequence_dataset`, `가격예측.split_dataset` |
| 6 | `가격예측/train_common.py` | v2 계열 공용 학습/평가 유틸(`train_transformer`, `evaluate_predictions` 등) | `가격예측.model` |
| 7 | `가격예측/train_v2.py` | v2 재설계 피처로 재학습(val만, test는 트랙1 결합 시점까지 보류) | `constants`, `가격예측.sequence_dataset`, `가격예측.split_dataset`, `가격예측.train_common` |

**진단 체인 (실험·진단용, val만 사용, test 미사용 — 순서대로):**

| 순서 | 파일 | 역할 | 의존성 |
|---|---|---|---|
| 진단1 | `가격예측/diagnose_baseline.py` | 비정상성(train/test 가격 레벨 괴리)·다중공선성 진단, 상관관계 히트맵 | `constants`, `가격예측.sequence_dataset`, `가격예측.split_dataset`, **`가격예측.train`**(train_common 아님) |
| 진단2 | `가격예측/compare_v2_variants.py` | 선형회귀 vs Transformer 소형 vs 기존 3-way 비교 | `constants`, `가격예측.sequence_dataset`, `가격예측.split_dataset`, `가격예측.train_common` |
| 진단3 | `가격예측/diagnose_down_class.py` | 하락 클래스 precision/recall/PR curve, 게이트 기준(precision +2%p, recall≥0.5) 확정 | 위와 동일 + `가격예측.train_common` |
| 진단4 | `가격예측/seed_stability_check.py` | 7시드 재현성 — "소형 TF가 baseline 돌파"가 우연이었음을 반증 | `constants`, **`가격예측.diagnose_down_class`**, `가격예측.sequence_dataset`, `가격예측.split_dataset`, `가격예측.train_common` |
| 진단5(최종) | `가격예측/pr_curve_vs_random.py` | PR curve vs 무작위 sanity check — **신호 부재 최종 확정** | `constants`, `가격예측.sequence_dataset`, `가격예측.split_dataset`, `가격예측.train_common` |

**결론**: 가격+거시지표 단독으로는 익일 방향성이 무작위/baseline과 구분 안 됨 (RMSE/MAE는 v2에서
개선). 스트레스 구간 홀드아웃 평가(체크포인트5)는 **미착수** — 정상 레짐에서부터 신호 부재가
확정돼 우선순위 밀림, 재개 여부 사람 결정 대기.

**별도 도구 (파이프라인과 의도적으로 분리):**

| 파일 | 역할 | 상태 |
|---|---|---|
| `analysis/detect_stress_period.py` | 스트레스 구간 변곡점 탐지(이동평균 |등락률| 기반). `constants.py`의 상수를 사람이 수동 확정할 때만 재실행 | 일회성 도구. **자동 재탐지 금지**가 설계 원칙 — 학습 파이프라인은 이 스크립트를 호출하지 않고 `constants.py` 상수만 읽음 (의존: `db_manager`) |

### 트랙 3 — 뉴스/감성 (Task A~F, Task B 방향 이번 세션에 재개)

2026-08-23 구조 정리로 `NaverSearchBackfill.py`가 공통/최초적재/일일수집 3개 파일로 분리됨.
날짜 상수는 `constants.py`(`NEWS_BACKFILL_START`/`NEWS_BACKFILL_END`, 2023-08-23~2026-08-23
고정값)로 중앙화. 뉴스_일일수집.py는 이번에 신규 정의된 개념(다른 로더와 같은 "DB 최신일+1 ~
오늘" 패턴) — 공백이 14일을 넘으면 자동 캐치업하지 않고 최초적재 사용을 안내한다. 14일 기준은
2026-08-23 실측(조기종료 휴리스틱 적용 후 일평균 19.47초/일, 14일≈4.55분 — 무인 job 예산으로
적절)으로 근거를 확정함(사람 승인, 상세는 `뉴스_일일수집.py`의 `MAX_CATCHUP_DAYS` 주석 참고).

| 파일 | 역할 | 상태 |
|---|---|---|
| `뉴스데이터/뉴스_공통.py` | search.naver.com 크롤링 공용 상수·함수(`crawl_day`, `fetch_search_page`, `passes_quality_filter`, `upsert_articles`, `clean_title`/`clean_press` 등) | **핵심.** `clean_title`/`clean_press`/`upsert_articles`는 `NaverFinanceNews.py`에서 이관 — 살아있는 코드가 폐기 파일에 의존하던 역방향 구조를 바로잡음. 2026-08-29: 예방적 쿨다운 추가 — `fetch_search_page` 누적 호출이 80회에 도달할 때마다 90~120초 대기(`PREVENTIVE_COOLDOWN_EVERY`/`_RANGE`), 기존 회로차단기(`CONSECUTIVE_FAILURE_LIMIT`/`BACKOFF_SECONDS`)와는 별개로 얹은 사전 예방 조치. 모듈 전역 카운터라 `run_backfill_resumable`이 날짜를 넘나들며 호출해도 하루 단위로 끊기지 않고 누적됨. 의존: `db_manager` 없음(순수 크롤링/텍스트 유틸) |
| `뉴스데이터/뉴스_최초적재.py` | 명시적 날짜범위(기본 `NEWS_BACKFILL_START`~`END`) 대량 백필, 체크포인트 기반 재개 | **핵심(현재 승인된 유일한 백필 소스, 2026-08-23 전면 백필 승인)**. 의존: `constants`, `db_manager`, `뉴스데이터.뉴스_공통` |
| `뉴스데이터/뉴스_일일수집.py` | 종목별 DB 최신 수집일(`source='search_backfill'` 기준)+1 ~ 오늘 캐치업. 공백 14일 초과 시 자동 캐치업 안 하고 경고 | **핵심(신규, 아직 스케줄러 연결 전 — 로드맵 "이후" 단계에서 자동화 예정)**. 의존: `db_manager`, `뉴스데이터.뉴스_공통` |
**삭제 완료(2026-08-23)**: `NaverFinanceNews.py`(finance.naver.com 종목뉴스, `source='finance_crawl'`
— 페이지네이션 약 1주일 한계로 백필 부적합해 폐기, `clean_title`/`clean_press`/`upsert_articles`는
`뉴스_공통.py`로 이관 완료), `NaverSearchBackfill.py`(위 3개 파일로 분리된 원본), `NaverNews.py`
(네이버 뉴스 검색 API로 당일 뉴스만 수집하던 파일, DB 기여 0행이라 폐기), `NaverNewsCrawl.py`
(네이버 뉴스 검색 웹페이지 크롤링으로 변동성 임계값 초과일의 D-1·D 이틀치를 수집하던 파일,
발행시각 미수집·UI텍스트 혼입 등 결함으로 뉴스_공통/최초적재/일일수집으로 대체 완료), `sentiment_score_label.py`
(CLAUDE.md D-1이 폐기한 구 5-tier 절대임계값 라벨 체계 그 자체 — **Task E, 초과수익률+롤링표준화
구현 전까지 대체 라벨 소스가 없다는 점에 주의**). 삭제 전 grep으로 다른 파일의 실제 import 의존
없음을 확인했다.
| `감성분석/kobert_dataset.py` | `daily_news` 로더+split+Dataset | 핵심이나 **현재 `sentiment_score`(폐기 대상 라벨) 컬럼에 의존** — Task E 완료 전엔 실질 사용 불가. 의존: `db_manager` |
| `감성분석/baseline_tfidf.py` | TF-IDF+로지스틱회귀 진단 베이스라인 | 실험·진단용(Task A에서 1회 사용). 의존: `감성분석.kobert_dataset` |
| `감성분석/kobert_train.py` | KoBERT 파인튜닝 | ⚠️ **재실행 금지**(데이터 재구축 전까지, CLAUDE.md 기 명시). 의존: `감성분석.kobert_dataset` — import 방식을 `from 감성분석.kobert_dataset import ...`로 수정 완료(2026-08-23, 다른 파일들과 관례 통일). `import 감성분석.kobert_train`으로 ImportError 없음 확인(단 `__main__` 블록은 재학습을 바로 시작하므로 재학습 금지 원칙에 따라 실제 실행으로는 검증 안 함) |

### 트랙 3 부가 — BigKinds 데이터 소스 (2026-08-29 신규, Task A~F 체계 미편입)

`daily_news`(search_backfill 계열)와는 완전히 분리된 별도 실험 소스. Task A~F 어느 단계에도 아직
공식 편입되지 않았고, 기존 뉴스/감성 트랙과 어떻게 관계지을지(대체/보완/별도 비교)는 사람이 아직
결정하지 않았다.

| 파일 | 역할 | 상태 |
|---|---|---|
| `빅카인즈/빅카인즈_적재.py` | `빅카인즈_뉴스/`(gitignore됨) 폴더의 BigKinds CSV/Excel 일괄 적재 → `daily_news_bigkinds` | 최초 적재 완료(20,053행, 2026-08-29). `뉴스 식별자` dtype 버그 수정 반영됨(위 DB 스키마 절 참고). 의존: `db_manager` |

옛 `쓰래기통/Bigkinds.py`(daily_news 대상, CSV 전용, 레거시)를 사람이 되살려 참고했지만 재사용하지
않고 새로 작성했다 — 테이블도 별도, 컬럼도 더 풍부하다(본문 전체·URL·카테고리 등). `쓰래기통/Bigkinds.py`
자체는 여전히 미사용 상태로 `쓰래기통/`에 남아 있다.

### 기타

| 파일 | 역할 | 상태 |
|---|---|---|
| `쓰래기통/Bigkinds.py` | BigKinds CSV 일괄 적재(daily_news 대상, 레거시) | 폐기 후보(기 확정, 격리 완료). 2026-08-29 참고용으로 복원됐으나 재사용 안 함 — 위 `빅카인즈/빅카인즈_적재.py` 참고 |

### 감성 통합 실험의 스코프 (2026-08-23 명시)

**감성 피처 통합 실험은 뉴스 커버리지(`NEWS_BACKFILL_START`~`NEWS_BACKFILL_END`,
2023-08-23~2026-08-23, 약 3년)로 제한되며, 이는 기존 T-1(가격+거시지표 전용, 2020~2026 전체)
학습 범위와 다르다 — 서로 다른 서브 실험으로 명확히 구분한다.** 향후 "가격+거시 vs 가격+거시+뉴스"
최종 비교(T-1 문서의 test 평가 보류 사유 참고)를 할 때, 두 모델의 학습/평가 구간을 뉴스 커버리지
범위로 맞출지 아니면 T-1 원 구간을 유지하고 뉴스 쪽만 3년치로 제한된 서브셋 비교를 할지는
그 시점에 사람이 결정한다.

### 이번 감사로 드러난 사항 — 처리 현황 (2026-08-23)

- ✅ `NaverFinanceNews.py` 폐기 확정 — 위 표에 반영 완료
- ✅ `kobert_train.py`의 bare import(`from kobert_dataset import`)를 `from 감성분석.kobert_dataset import`로 수정. `import 감성분석.kobert_train`으로 ImportError 없음 확인(단, `__main__` 블록은 재학습을 즉시 시작하므로 재학습 금지 원칙에 따라 실제 학습 실행으로는 검증하지 않음 — import 단계만 확인)
- ⏸️ **"프로젝트 구조" 트리(`analysis/`, `가격예측/` 미반영) 갱신은 보류로 확정(2026-08-23, 사람 결정).**
- ✅ **날짜 상수 중앙화 + 초기적재/일일수집 파일 분리 완료(2026-08-23)** — `constants.py`에
  `NEWS_BACKFILL_START/END`, `STOCK_INITIAL_LOAD_START` 추가. 주가/시장지표/뉴스 세 트랙 모두
  "초기적재"와 "일일수집"을 별 파일로 분리(뉴스_일일수집.py는 신규 개념). 옛 파일 8개
  (`FinanceData_load.py`, `market_index_load.py`, `ecos_load.py`, `NaverSearchBackfill.py`,
  `NaverFinanceNews.py`, `NaverNews.py`, `NaverNewsCrawl.py`, `sentiment_score_label.py`)는
  grep으로 다른 파일의 실제 import 의존이 없음을 확인한 뒤 삭제 완료(2026-08-23). 전 신규 파일
  import 검증 완료(실제 데이터 적재 실행은 안 함).
- ✅ **주가/시장지표 재작업: 진짜 기능적 분리로 통일(2026-08-23)** — 처음엔 `주가_초기적재.py`가
  `주가_일일수집.py`의 함수를 직접 import하는 종속 구조였는데(이름만 분리, 실제로는 계층 구조),
  뉴스 트랙의 `뉴스_공통.py` 패턴에 맞춰 `주가_공통.py`/`지표_공통.py`를 신설하고 초기적재/일일수집
  둘 다 공통 파일만 참조하도록 재작업. 초기적재↔일일수집 상호 import 없음을 grep으로 교차 확인.
  백필 속도 개선(조기종료 휴리스틱, 서버 필터 확인)은 이 구조 정리 이후 별도 진행 예정.
- ✅ **테스트 누적 데이터 정리 + 백필 속도 개선 완료(2026-08-23)**
  - `daily_news_backup`(구 스키마 스냅샷, 5,278행) 삭제, `daily_news.source='search_backfill'`
    (그동안 소스비교/1개월/2023-09 검증 등으로 누적된 1,099행) 삭제 — legacy/finance_crawl은 그대로
    유지. 체크포인트 파일도 함께 삭제해 DB와 일관성 확보.
  - 언론사 서버사이드 필터(`news_office_checked`/`office_type`/`office_category`) 실측 확인 —
    **작동 안 함**(파라미터를 채워 요청해도 결과가 필터링되지 않음, 죽은 기능으로 판단). 클라이언트
    측 필터(품질 필터)에 계속 의존.
  - `crawl_day()`에 조기 종료 휴리스틱 추가: 연속 5페이지 또는 누적 50건 후보가 필터 통과 0건이면
    그 날짜를 조기 종료(`early_stopped` 플래그). 구현 중 `degraded`(응답 이상 의심) 판정이 이 정상
    조기종료 케이스까지 오분류하던 버그를 함께 고쳐 `page_errors`가 실제로 있을 때만 `degraded`가
    걸리도록 조건을 좁힘.
  - 1주일 실측(005930, 2026-08-17~08-23, 조기종료 적용 후): **136.3초, 일평균 19.47초**. 3년
    전체(`NEWS_BACKFILL_START`~`END`, 1,096일) 예상 약 5.9시간.
  - **확정(2026-08-23, 사람 결정)**: `NEWS_BACKFILL_START`는 3년(2023-08-23~2026-08-23) 그대로
    유지. 애초에 세웠던 "30분~1시간 예산" 기준은 철회 — 5.9시간은 사람이 지켜봐야 하는 시간이 아니라
    체크포인트 기반 중단/재개가 이미 검증된 백그라운드 작업이므로 문제 없다고 판단. 기간을
    3~6개월로 줄이면 val/test 커버리지가 손상되므로 원래 목표(뉴스 커버리지로 val+test 전체 커버)를
    지키는 쪽 선택. **실제 3년 전체 백필은 사람이 여러 세션에 나눠 직접 실행 예정**
    (`python -m 뉴스데이터.뉴스_최초적재 --tickers 005930 --start 2023-08-23 --end 2026-08-23`,
    체크포인트 기본 경로 `뉴스데이터/checkpoints/search_backfill_progress.csv`).
  - `MAX_CATCHUP_DAYS=14`(뉴스_일일수집.py) 확정 — 근거는 위 실측(14일×19.47초≈4.55분).

---

## 알려진 이슈 / 정리 필요 항목
- `requirements.txt`에 `psycopg`와 `psycopg-binary` 중복 — 정리 검토
- `market_indicators.unit` DEPRECATED — 적재 시 NULL로 두고 `indicator_meta.unit` 사용
- `쓰래기통/` 폴더명 오타는 의도된 것("안 쓰는 코드 보관용")

---

## 로드맵

### 주가/거시 트랙 (Task G) — ✅ 전체 완료
| 단계 | 내용 | 상태 |
|---|---|---|
| G-0 | 스키마 확인 | ✅ 완료 |
| G-1 | 스키마 설계 + 마이그레이션 + 제약 검증 | ✅ 완료 |
| G-2 | `시장지표/market_index_load.py`, KOSPI/KOSDAQ/USD_KRW 적재 | ✅ 완료 (14,072행) |
| G-3 | ECOS 지표 코드 조회 → 적재 (기준금리·국고채3·10년·CPI·M2·선행지수) | ✅ 완료 (24,883행) |
| G-4 | `feature_loader.py` — 누수 없는 피처 조회 + 검증 | ✅ 완료 |

G-2/G-3 완료로 KOSPI 등락률과 거시지표 6종이 확보되어 초과수익률 라벨링(Task E)이 열린다. G-4 완료로 Transformer ablation baseline 착수에 필요한 피처 조회 함수도 준비됐다.

### 뉴스/감성 트랙 (Task A~F) — Task B 방향 재개, 백필 진행 중

| 단계 | 내용 | 상태 |
|---|---|---|
| A | 원인 확정 진단 | ✅ 완료, 판정 승인 |
| B | 수집기 교체 → search.naver.com 날짜범위 크롤러(`뉴스_최초적재.py`/`뉴스_일일수집.py`/`뉴스_공통.py`)로 확정(2026-08-23, finance.naver.com은 페이지네이션 한계로 폐기). 발행 "시각"은 이 소스로 확보 불가 — Task D가 안 B(완화)로 대체 확정됨 | ✅ 방향 확정, **전면 백필 실행 중** (아래 참고) |
| C | 전면 재수집. 종목은 당분간 005930 단일(2026-08-23 사람 결정, SK하이닉스 드랍). 섹터 분산 확장은 이후 재검토 | 진행 중(B의 백필과 사실상 통합) |
| D | 이벤트 윈도우 재정렬 — **안 B(완화): D-1 09:00~D 09:00 24시간 윈도우로 확정**(2026-08-23, search_backfill이 시각 정보 없음) | 확정, **착수는 백필 완료 후**(아래 참고) |
| E | 라벨 생성 — D-2 롤링 z-score, 방향(5/3-class)·변동성(2-class) 라벨. 상세는 `TASK_EF_라벨링_비교실험.md` 참고(2026-08-29 재작성) | 보류(백필 완료 후 착수) |
| F | 비교 실험 — KR-FinBERT(KoBERT 아님), 데이터소스 교차평가(크롤링/빅카인즈) 포함. 상세는 `TASK_EF_라벨링_비교실험.md` 참고 | 보류 |

#### 뉴스 백필 진행 상황 (2026-08-29 갱신)

**실행 중.** `python -m 뉴스데이터.뉴스_최초적재 --tickers 005930 --start 2023-08-23 --end 2026-08-23`을
사람이 여러 세션에 나눠 직접 실행 중(2026-08-29 세션 동안에도 백그라운드로 계속 진행됨 — python
프로세스 실행 확인됨). 체크포인트(`뉴스데이터/checkpoints/search_backfill_progress.csv`, git 미추적)
기준 스냅샷:
- 완료: 2023-08-23 ~ 2025-12-15 (846일 / 전체 1,097일 ≈ **77.1%**, 남은 251일) — 계속 갱신되는
  값이므로 다음 세션 시작 시 체크포인트 파일로 재확인할 것
- 2026-08-29: `뉴스데이터/뉴스_공통.py`에 예방적 쿨다운 추가(위 파이프라인 표 참고) — 403 유발
  자체를 줄이려는 조치. 일평균 신규 삽입 건수가 개선됐는지는 아직 재검증 안 함

**다음 세션 시작 시 할 일**:
1. 체크포인트 파일(`뉴스데이터/checkpoints/search_backfill_progress.csv`)로 실제 진행률 재확인
   (마지막 완료 날짜, 전체 대비 %)
2. 백필이 멈춰 있다면(연속 2일 차단 등으로 스스로 중단됐을 수 있음) **다른 크롤러 프로세스가
   실행 중이 아닌지 먼저 확인**한 뒤, 같은 명령으로 재개:
   `python -m 뉴스데이터.뉴스_최초적재 --tickers 005930 --start 2023-08-23 --end 2026-08-23`
   (체크포인트에 없는 날짜부터 자동으로 이어짐)
3. 차단(🛑)이 자주 뜬다면 `뉴스데이터/뉴스_공통.py`의 `REQUEST_DELAY_RANGE`(현재 2.0~4.0초)를
   늘리는 것을 고려할 것

**백필 완료 후 QA 필요 사항**:
- **"403은 있었지만 kept>0라서 완료 처리된 날짜" 별도 점검** — `degraded` 판정은 `kept==0`일
  때만 걸리므로, 그 날 일부 페이지가 403이었지만 다른 페이지에서 몇 건 건졌다면 정상 완료로
  체크포인트에 기록된다. 이런 날은 **실제로는 그 날 관련 기사 일부를 놓쳤을 가능성**이 있다 —
  체크포인트에는 403 여부가 기록되지 않으므로, 완료 후 전체 실행 로그(있다면)에서 이런 날짜를
  추출해 재검증이 필요한지 판단할 것
- 완료 후 일평균 신규 건수를 2026-07/2023-09 검증치(중앙값 15~18건)와 비교 — 지금까지
  구간(중앙값 10건)이 계속 낮게 나오면 조기종료 휴리스틱이 너무 공격적인 건 아닌지 재검토

**Task B 재수집 완료 후 확인 필수 (기존, Task B 착수 전 조건에서 이관)**: 네이버 금융 종목뉴스
페이지의 과거 조회 가능 기간은 이미 실측 완료(약 4~7일, 부적합 확정) — 이 조건은 해소됨.

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
- `Database/{도메인}/테이블_삭제.sql` 실행 시 해당 도메인 테이블 DROP — 반드시 확인 후 실행
- **스키마 변경은 운영 DB에 `ALTER TABLE`로 적용하고, 같은 내용을 `Database/{도메인}/테이블_생성.sql`에도
  반영해 그 파일이 항상 "지금 스키마"를 그대로 재현하도록 유지한다(2026-08-23부터, 옛
  `Database/migrations/` 방식 폐지). 변경 사유는 커밋 메시지에 남긴다. 기존 테이블을 DROP하지 않는다**
- **기존 컬럼 값을 파괴적으로 덮어쓰지 않는다. 새 라벨·새 시각은 새 컬럼에 추가해 비교 가능하게 둔다**
- 모든 스크립트는 프로젝트 루트에서 `python -m <패키지>.<모듈>` 형태로 실행 가능해야 한다
- **모든 성능 비교 baseline은 해당 split의 실제 최빈 클래스 기준** (현재 test 31.8% / macro F1 0.0965)
- **예외를 `print`만 하고 삼키지 않는다.** 단위별 격리 + 최종 요약에 실패 항목 명시 + 0건 항목 경고
- 각 Task의 정지 지점에서 임의 판단으로 다음 단계에 진입하지 않는다
