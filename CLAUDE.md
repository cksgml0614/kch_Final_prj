# kch_Final_prj

주가 데이터 + 뉴스 데이터를 결합해 감성분석 모델(KoBERT)과 가격 예측 모델(Transformer)을 학습시키는 졸업작품 프로젝트.

## 지금 어디까지 왔는가 (세션 시작 시 먼저 읽을 것)

프로젝트는 **두 개의 독립 트랙**으로 병행 진행 중이다.

| 트랙 | 지시서 | 상태 |
|---|---|---|
| **뉴스/감성** (Task A~F) | `TASK_개정판_데이터_재구축.md`(A~D) + `TASK_EF_라벨링_비교실험.md`(E~F, 2026-08-29 분리) | Task A 완료, 판정 승인됨. Task B 완료 — 전면 백필 완료(2023-08-23~2026-08-28) + QA 검증·클러스터 확장 재크롤링·갭 구간 표본 재크롤링까지 전부 완료(`source='search_backfill'` 최종 14,392행). **Task E 완료 — 임계값·윈도우 확정(상세: `결과_TaskE_라벨링.md`)** |
| **주가/거시** (Task G) | `TASK_G_market_indicators_적재.md` | **G-0~G-4 전체 완료** |

**현재 상태(2026-09-01)**: Task G 완료 후 열렸던 두 갈래(Transformer 착수 / 뉴스·감성 Task B 재개) 모두
착수됐다. **Task T-1(Transformer ablation baseline)은 완료돼 "가격+거시지표만으로는 익일 방향성이
무작위 수준과 구분 안 됨"으로 확정**됐다(상세는 "파이프라인 실행 순서 > 트랙 2" 참고). **뉴스/감성
Task B는 전면 백필 완료 후 QA 검증(53일 재크롤링) + 클러스터 확장 재크롤링(91일) + 갭 구간
표본 재크롤링(47일, 부분 확인)까지 모두 마쳤다** — `source='search_backfill'` 14,077 → 최종
**14,392행**. QA에서 드러난 구조적 누락은 2024-02와 2024-07~08에 국한된 것으로 판단됐다(갭
구간 표본 검증 결과 3년 전체로 일반화되지 않음이 재확인됨). 상세는 로드맵 절의 "뉴스 백필 완료
+ QA 결과" 참고. **Task E(라벨 생성)도 완료됐다** — `daily_labels`(z_score 4,902행) 적재, 윈도우
20/60/120 전부·방향 5/3-class 전부·변동성 \|z\|>1.0으로 임계값 확정. 1차 실험의 분포 이동
문제(train 중립 50.4%→test 극단 31.8%)가 해소됐음을 확인했다. 상세 수치는 `결과_TaskE_라벨링.md`,
확정 사항 요약은 D-2 참고. **Task F 게이팅·검증도 완료됐다**(2026-09-01~09-02) — 뉴스 라벨
매칭에서 당일 매칭 누수를 발견해 D+1 거래일 매칭으로 수정했으나(Task D "안 B"의 실제 구현),
셔플 테스트·클래스 사전분포 무작위 예측기로 재검증한 결과 익일(h=1) 라벨은 신호가 없었다(p전부
≥0.258). 주간 단위 누적 라벨(h=3/5/10)로 재설계해도 결론은 같았다(p전부 ≥0.387) — **뉴스
텍스트와 초과수익률 사이에 검출 가능한 관계 없음**으로 확정. 이후 절대수익률 라벨·키워드
관련성 필터로도 신호가 나오지 않았고, **가장 성능이 나았던 조합을 KR-FinBERT로 실제
파인튜닝한 결과도 사전분포 무작위 예측기보다 낮은 성능으로 확인돼 투입 보류가 확정**됐다
(2026-09-06). 상세는 로드맵 "Task F 게이팅·검증·horizon 확장 실험" 및
`결과_TaskF_게이팅검증.md` 참고.

**추가 갱신(2026-09-06)**: Task T-1의 "무작위 수준" 결론 이후, 방향(상승/하락) 예측을 8종목
pooled+종목임베딩으로 재시도했으나 역시 무작위 수준으로 재확인돼 **방향 예측 트랙을 완전히
폐기**하고 **변동성(|수익률|) 예측으로 전환**했다(상세: `결과_TaskT_방향예측_폐기.md`). 전환
직후 GARCH(1,1)가 SMA20을 이긴다는 결과를 얻었으나, 이후 GARCH 계산 함수의 날짜 라벨링
버그(그날 자신의 수익률이 이미 섞여 있던 미래정보 누수)를 발견해 수정했고, **수정 후
재검증한 결과 GARCH·하이브리드 Transformer 둘 다 SMA20을 못 이기고 조기경보 능력도
우연 수준으로 무너짐**이 확정됐다. 이후 실험한 Parkinson(고가-저가 범위) 기반 변동성
추정량만이 유일하게 GARCH·SMA20을 실제로 이기는 baseline으로 확인됐으나, 이를 Transformer
피처로 추가하거나 압축 피처 없이 원본 시퀀스를 직접 줘도 성능이 개선되지 않았다 — 최근
진단으로 이 실패가 "복잡한 과적합"이 아니라 "학습 초반(첫 epoch)에 거의 상수 예측으로
수렴해버리는 분산 붕괴"임이 확인됐다.

**50종목 확장(2026-09-06)**: "표본이 적어(8종목) 분산 붕괴가 난다"는 가설로 종목을
시가총액 상위 **50개**(우선주 제외, `fdr.StockListing('KOSPI')` 기준)로 확장 — 이 과정에서
`034730`이 "현대건설"로 잘못 등록돼 있던 것을 발견해 "SK"(주식회사, 지주)로 정정(가격
데이터 자체는 처음부터 정확했고 이름표만 틀렸던 것, `constants.py`에는 이미 정정 반영됨).
**50종목 pooled 하이브리드 모델이 GARCH·SMA20 baseline을 통계적으로 유의하게 이겼다**
(N=30 셔플 검증, p=0.0323, 30/30 전부 실제 모델이 우세) — 분산 붕괴도 뚜렷이 완화됨
(예측 std/target std 비율 9.5%→30.8%). Parkinson-SMA20 대비는 원본 RMSE만 근소하게 미달.

**종목평균 제거(demean) 분해 검증 + 100종목 확장 + 최종 확정(2026-09-07)**: "분산 확대가
종목 간 평균 수준 차이만 반영한 것 아니냐"는 우려를 검증하기 위해 종목별 평균을 뺀 잔차로
상관계수·RMSE를 재계산 — demean 후에도 상관계수가 유지되고(r 0.3434→0.2556), Parkinson-
SMA20 대비 원본 RMSE의 근소한 열세도 demean에서는 사라짐을 확인해 **우려는 기각**됐다.
이 두 효과(분산 붕괴 완화, demean 후에도 살아있는 예측력)가 표본을 더 늘려도 이어지는지
확인하기 위해 종목을 50→**100개**로 추가 확장(대체 포함, `결과_TaskT_변동성예측_최종.md`
[13] 참고) — **100종목에서 GARCH·SMA20·Parkinson-SMA20 세 baseline을 원본·demean 지표
양쪽에서 전부 통계적으로 유의하게 이기는 것으로 최종 확정**됐다(예측 std/target std
비율 33.4%, demean r=0.2795, N=5 셔플 격차 90~120σ). **다만 이 결과에는 정직한 한계가
있다** — demean r=0.2795는 R²≈7.8%로 설명력 자체는 약하고, 예측 분포가 극단치(실제
target 최대 26.24 vs 예측 최대 4.52)를 거의 못 잡으며, walk-forward 재검증과 실제 의사결정
백테스트가 전혀 없다. **"통계적으로 유의한 개선"이지 "실전 투입 가능한 수준"은 아직
아니다.** 100종목 검증 결과를 그대로 옮긴 자동화 파이프라인(`가격예측_변동성_공통.py`/
`가격예측_변동성_일일수집.py`/`종목_월간갱신.py`, 배포 게이트는 GARCH·SMA20·Parkinson-
SMA20 셋 다 이겨야 통과로 확정)도 **코드 작성까지 완료했으나 실행/스케줄러 등록은 하지
않았다**. 시간순 전체 경과·모든 수치표는 `결과_TaskT_변동성예측_최종.md` 참고.

**다음 세션 시작 시 할 일**:
1. `Database/가격예측/테이블_생성.sql` 하단 마이그레이션(`parkinson_sma20_baseline`/
   `gate_vs_parkinson` 컬럼 추가)을 운영 DB에 `ALTER TABLE`로 적용
2. `가격예측_변동성_일일수집.py` 최초 1회 수동 실행해 정상 동작 확인(GARCH 파라미터 캐시가
   비어있어 첫 실행은 100종목 전부 재적합 — 이후 실행부터 캐시 재사용으로 빨라짐)
3. 크론/스케줄러 등록(`종목_월간갱신.py`는 기본 `--dry-run` — 실제 반영은 `--apply` 필요)
4. walk-forward(여러 시점 반복 분할) 재검증 — 지금까지는 고정 단일 70/15/15 분할 하나로만
   검증했음(위 "정직한 한계" 참고)
5. 무인 실행 안정성 확보: Windows 작업 스케줄러의 Wake Timer 설정(절전 중 PC를 깨워
   실행) + 실행 성공/실패를 화면 녹화 등으로 확인할 준비

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

**✅ 2026-08-30 확정(Task E 실행 결과 기반, 2026-08-31 구현·검증)**: 윈도우 N=20/60/120
**전부 사용** — 세 윈도우의 z 분포·라벨 분포 차이가 **±1%p 이내**로 미미해 사전에 하나로 좁힐
근거가 없었다. Task F에서 윈도우 3종을 그대로 비교 축으로 유지해 모델 성능으로 최종 판단한다.
방향 5-class(z ±0.5/±1.5)와 3-class(z ±0.5) **둘 다 사용**. 변동성 2-class는 **\|z\| > 1.0**
채택 — `>0.5`는 고변동 비율이 58~61%로 다수 클래스가 되어 변별력이 약하고, `>1.5`는 12% 안팎으로
줄어 표본이 지나치게 작다. `>1.0`은 정상/고변동이 대략 70/30으로 갈려 두 극단의 중간에서 가장
실용적인 균형을 보였다.

⚠️ **5-class 극단 클래스(-2) 표본 부족 주의**: train 5.1~5.8%, val 0.9~1.8%(윈도우별)로 표본이
극히 적다. 3-class와 함께 계속 사용하되, Task F에서 이 클래스의 precision/recall이 낮게 나오거나
5-class 전체 성능이 저조해도 **"모델 한계"로 먼저 결론짓지 말고 "표본 부족"을 우선 의심할 것** —
특히 val 기준 게이팅·early stopping 지표를 볼 때 이 점을 감안해야 한다.

상세 근거·전체 분포표는 `결과_TaskE_라벨링.md` 참고.

### D-3. 학습 기간 — 전체 기간 사용 (2026-08-30 개정, 옛 "정상 레짐/스트레스 2분할" 폐기)

기존 방침(정상 레짐 2023-09~2026-03을 주 실험에, 스트레스 2026-04~2026-07을 별도 케이스
스터디로 분리해 주 실험 성능에서 제외)을 **폐기**한다. **전체 기간을 그대로 학습에 사용**한다
— 종목별 스트레스 구간 자동 탐지·제외 로직 없이 시계열 그대로 D-8의 70/15/15 분할을 적용한다.

개정 사유:
- **Task C 종목 확장과의 충돌**: 종목을 확장하면 종목마다 스트레스 구간이 달라 매번 탐지·제외하는
  로직이 필요해져 자동화가 복잡해진다
- **미래 정보 누수**: 실제 운영에서는 스트레스 구간의 시작을 사전에 알 수 없다. 사후적으로만
  판별 가능한 정보(변동성이 이미 커진 뒤에야 "여기가 스트레스 구간"이라고 알 수 있음)로 학습
  데이터를 걸러내는 것은 일종의 미래 정보 누수다
- **D-2 롤링 z-score가 레짐 변화를 이미 흡수함**: `sigma_t`가 직전 N거래일 기준으로 매번
  갱신되므로, 폭락장에서는 sigma가 커져 같은 -5%도 z값이 작아진다. 이것이 1차 실패 원인이었던
  고정 절대 임계값과 결정적으로 다른 점이다 — 고정 임계값 체계에서는 이 흡수가 불가능했다
- **성능 저하는 오염이 아니라 분석 결과**: 스트레스 구간에서 성능이 떨어진다면 그것은 데이터를
  제외해야 할 오염이 아니라 D-4가 다룰 케이스 스터디 그 자체다

**단, D-8의 split별 레짐 특성 3종 보고는 그대로 유지한다** — test 구간이 스트레스 기간을
포함한다는 사실이 숫자로 드러나야 결과 해석이 가능하다.

**✅ 위 흡수 효과가 Task E 실행으로 정량 확인됨(2026-08-31)**: split별 D-8 레짐 특성에서
**KOSPI 자체의 평균 변동성은 train→test 구간에서 약 3.7배**(0.867%→3.188%) 뛰지만, **초과수익률의
평균 변동은 약 1.7배 증가에 그친다**(0.993%→1.684%). 시장 전체가 흔들린 폭의 상당 부분이
초과수익률 계산에서 이미 상쇄된다는 뜻 — "롤링 z-score가 레짐 변화를 흡수한다"는 위 논거가
가설이 아니라 실측으로 뒷받침됨. 전체 표는 `결과_TaskE_라벨링.md` [4] 참고.

⚠️ 이 개정은 **뉴스/감성 트랙(Task E/F) 한정**이다. `constants.py`의 `STRESS_PERIOD_START/END`는
애초에 Task T(가격+거시지표 트랜스포머 베이스라인) 전용 상수로 명시돼 있었고, 이번 개정과
무관한 별개 트랙·별개 결정이다(상세는 아래 `constants.py` 상수 배경 절 참고).

상세 반영은 `TASK_EF_라벨링_비교실험.md`의 "학습 기간 — 전체 기간 사용" 절 참고.

### D-4. 스트레스 구간 성능 분석 — 사후 슬라이싱 (D-3 개정에 따른 방법 변경, 2026-08-30)

D-3 개정으로 스트레스 구간이 더 이상 별도 홀드아웃이 아니라 주 실험 데이터에 포함되므로,
"별도로 모델을 스트레스 구간에 적용"하는 방식이 아니라 **통합 test 셋 내에서 날짜 기준으로
사후 슬라이싱**해 정상/스트레스 구간별 성능을 비교한다: 성능 저하 폭, 무너지는 클래스, 예측
실패일과 서킷브레이커·사이드카 발동일 중첩 여부를 분석.

결론 방향(변경 없음): "뉴스 감성 예측은 정보가 가격을 주도하는 국면에서 작동하며, 수급·기계적 청산이 주도하는 국면에서는 원리적으로 작동하지 않는다." 선행연구(Antweiler & Frank 2004; Tetlock 2007~2011)의 경계 조건을 실증하는 사례가 된다.

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
주 실험은 **전체 기간**(D-3 개정, 2026-08-30 — 정상 레짐/스트레스 분리 폐기) 내 70/15/15 시계열
분할. **각 split의 레짐 특성(전 거래일 평균 |cr|, 평균 |초과수익률|, KOSPI 평균 |cr|)을 반드시
함께 보고.** Task F 시점에 walk-forward 검증 도입 여부 재논의.

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

⚠️ **2026-08-30 데이터 규모 정정**: `daily_news` 학습 가용량은 `source='search_backfill'`만
집계해야 한다(QA·클러스터 확장·갭 구간 재크롤링까지 완료한 최종값 **14,392건**) — 이전 세션
대화에서 언급된 "25,895건"은 `daily_news` 전체(legacy 10,618 + finance_crawl 1,200 +
search_backfill 14,077, 재크롤링 전 기준)의 합계였고 학습 정본 규모가 아니다.
`daily_news_bigkinds`(20,053건, 위 스키마 절 참고)가 이보다 약 1.4배 많다 — 위 (b) 대체 가능성
검증의 중요도가 그만큼 올라간다: 검증에 성공하면 학습 데이터를 약 40% 더 확보할 수 있다.

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
├── 결과_TaskE_라벨링.md              # Task E 실행 결과 상세 수치 (2026-08-31, CLAUDE.md는 요약·참조만)
├── 결과_TaskF_게이팅검증.md          # Task F 게이팅·검증·horizon 확장 실행 결과 (2026-09-02, CLAUDE.md는 요약·참조만)
├── TASK_G_market_indicators_적재.md  # 주가/거시 트랙 지시서 (Task G)
├── 결과_TaskT_방향예측_폐기.md        # Task T 방향예측 폐기 배경·판정 근거 (2026-09-06)
├── 결과_변동성_조기경보_검증.md        # ⚠️ 폐기됨(GARCH 라벨링 버그) — 실패 원인 추적용으로만 보존 (2026-09-06)
├── 결과_TaskT_변동성예측_최종.md      # 변동성 예측 트랙 전체 경과·수치표 정본, 진행 중 (2026-09-06)
├── Database/              # 2026-08-23 도메인별 폴더로 재구성 (옛 create_tables.sql/init_table.sql/migrations/ 삭제)
│   ├── 뉴스/               # 테이블_생성.sql(daily_news) / 데이터_삭제.sql / 테이블_삭제.sql
│   ├── 주가/               # 테이블_생성.sql(daily_stock_prices) / 데이터_삭제.sql / 테이블_삭제.sql
│   ├── 시장지표/            # 테이블_생성.sql(indicator_meta→market_indicators, FK 순서) / 데이터_삭제.sql / 테이블_삭제.sql
│   └── 라벨/               # 테이블_생성.sql(daily_labels, horizon_h PK 포함) / 데이터_삭제.sql / 테이블_삭제.sql (2026-08-30 신설)
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
├── 라벨/                  # Task E: daily_labels 적재·보고 (2026-08-31 신설)
│   ├── 라벨_공통.py       # z-score 계산 + 라벨 파생 함수 + 조회/split 유틸
│   ├── 라벨_생성.py       # daily_labels UPSERT (윈도우 x horizon)
│   └── 라벨_보고.py       # Task E 정지 지점 보고
├── 감성분석/
│   ├── kobert_dataset.py       # daily_news(라벨 有) -> KoBERT Dataset/DataLoader
│   ├── kobert_train.py         # 부분 freeze 파인튜닝 + early stopping
│   ├── baseline_tfidf.py       # TF-IDF + 로지스틱 회귀 baseline + 데이터 진단 지표
│   ├── taskf_gating.py         # Task F 게이팅(h=1) — 라벨 3종 x 윈도우 3종
│   ├── taskf_validate.py       # taskf_gating.py 결과 검증(셔플/사전분포/부트스트랩)
│   ├── taskf_gating_horizon.py # Task F 게이팅+검증(h=3/5/10, N=60 고정)
│   └── checkpoints/            # best_model.pt, training_history.png (git 미추적)
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

### daily_labels `(ticker, date, window_n, horizon_h, label_basis)` PK — `Database/라벨/테이블_생성.sql` (2026-08-30 설계 확정, 2026-08-31 생성·적재, 2026-09-02 horizon_h 추가, 2026-09-06 label_basis 추가)

Task E 라벨을 뉴스 테이블(`daily_news`/`daily_news_bigkinds`)에 컬럼으로 추가하지 않고 별도
테이블로 분리하기로 결정(2026-08-30, `TASK_EF_라벨링_비교실험.md` 실행 전 검토). 라벨은
`(ticker, date)` 단위인데 뉴스 테이블에 넣으면 같은 날 기사 수만큼 같은 값이 중복 저장되고,
임계값을 바꿀 때마다 `daily_news`/`daily_news_bigkinds` 두 테이블을 모두 갱신해야 해서 관리가
어려워진다.

```
ticker         varchar     NOT NULL   -- PK
date           date        NOT NULL   -- PK, 거래일 기준(앵커일 t)
window_n       integer     NOT NULL   -- PK, sigma 계산 윈도우(20/60/120 중 하나)
horizon_h      integer     NOT NULL   -- PK, 누적 수익률 기간(거래일). 1=익일(기존), 3/5/10=n거래일 누적(2026-09-02 추가)
label_basis    varchar     NOT NULL   -- PK, 'excess_return'(종목-KOSPI, 기존) / 'absolute_return'(종목 순수 change_rate, 2026-09-06 추가)
excess_return  numeric                -- t부터 horizon_h거래일(t..t+h-1) 누적 수익률 합(label_basis에 따라 초과/순수). h=1이면 당일 값과 동일
sigma          numeric                -- 직전 window_n개 앵커일의 동일 horizon_h/label_basis 누적값 표준편차(t 시점 미포함)
z_score        numeric                -- excess_return / sigma
```

- **`z_score`만 저장하고 라벨(방향 5-class/3-class, 변동성 2-class)은 저장하지 않는다** — 조회
  시 임계값을 적용해 파생한다. 임계값을 바꿔도 재계산이 필요 없고, 여러 임계값 조합을 자유롭게
  실험할 수 있다
- `window_n`/`horizon_h`를 PK에 포함해 윈도우 3종 x horizon 4종(1/3/5/10)을 한 테이블에 함께 보관
- 학습 시 `daily_news`/`daily_news_bigkinds`와 `(ticker, date)`로 조인한다. 뉴스 테이블에는
  라벨 컬럼을 추가하지 않는다. 기존 `daily_news.sentiment_score`(구 5-tier 라벨)는 D-1에 따라
  보존만 하며 이 신규 라벨과는 무관하다
- `Database/뉴스/` 등 기존 컨벤션대로 `Database/라벨/` 하위에 `테이블_생성.sql`/
  `데이터_삭제.sql`/`테이블_삭제.sql` 3종 작성 완료, 운영 DB에도 생성 완료(2026-08-31)
- 005930 x 윈도우 3종(20/60/120), 2020-01-02~2026-08-28 가격 이력 전체 대상으로 계산해
  4,902행 적재 완료(`라벨/라벨_생성.py`). `TRAIN_PERIOD` 구간 내 NULL 0건 확인(버퍼로 흡수)
- **2026-09-02 `horizon_h` 컬럼 추가**(ALTER TABLE로 운영 DB 적용, PK를 `(ticker, date,
  window_n)`에서 `(ticker, date, window_n, horizon_h)`로 확장) — 기존 h=1 행 4,902개는 값
  변경 없이 보존(재계산 결과가 기존 값과 완전히 동일함을 UPSERT의 `IS DISTINCT FROM`으로
  실측 확인). h=3/5/10 신규 14,706행 추가, 총 19,608행. 별도 테이블이 아니라 컬럼 추가를
  택한 이유: `window_n`을 이미 같은 테이블 PK에 포함시켜 다루던 기존 패턴과의 일관성
- **2026-09-06 `label_basis` 컬럼 추가**(ALTER TABLE로 운영 DB 적용, PK를 `(ticker, date,
  window_n, horizon_h)`에서 `(ticker, date, window_n, horizon_h, label_basis)`로 확장) —
  "시장 전체에 좋은 뉴스"로 인한 신호가 초과수익률의 KOSPI 차감으로 상쇄되는 것 아니냐는
  우려에서, KOSPI를 빼지 않은 순수 `change_rate` 기준 z-score를 `label_basis='absolute_return'`
  으로 추가 적재했다(기존 `label_basis='excess_return'` 19,608행은 값 변경 없이 보존,
  absolute_return 신규 19,668행 추가). `excess_return` 컬럼명은 그대로 재사용하고
  `label_basis`로 의미를 구분한다 — `horizon_h` 도입 때와 같은 컬럼 재사용 관례. **비교
  게이팅 실험 결과 absolute_return 라벨에서도 유의한 신호가 확인되지 않아 "KOSPI 차감으로
  시장 전체 신호가 상쇄된다"는 가설은 기각됐다** — 상세는 `결과_TaskF_게이팅검증.md` [5] 참고
- z 분포·라벨 분포·split별 비교·확정된 임계값/윈도우는 `결과_TaskE_라벨링.md`, Task F
  게이팅·검증·horizon·basis 비교 실험 결과는 `결과_TaskF_게이팅검증.md` 참고 — 상세 표는
  그 문서들이 정본이며 여기서는 중복 기재하지 않는다

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
- `load_labeled_news(ticker=None, table_name='daily_news', label_column='sentiment_score')` (2026-08-30 확장, 기존 호출부 하위 호환): `daily_news`/`daily_news_bigkinds` 공통 컬럼(`ticker,date,title,summary,press,article_url,target_date`)+`label_column`을 date 오름차순 로드
  - `table_name`은 화이트리스트(`{'daily_news','daily_news_bigkinds'}`), `label_column`은 안전한 식별자 정규식(`^[a-zA-Z_][a-zA-Z0-9_]*$`)으로 검증 후 SQL에 사용 — 둘 다 `%s` 파라미터화가 안 되는 식별자라 직접 검증 필요(SQL 인젝션 방지)
  - `label_column=None`이면 라벨 필터(`IS NOT NULL`) 없이 전체 로드 — 라벨이 아직 없는 소스(`daily_news_bigkinds`)를 Task E 착수 전 텍스트만 확인할 때 사용
  - `label_column='sentiment_score'`(기본값)일 때만 `LABEL_MAP = {-2:0,-1:1,0:2,1:3,2:4}`로 매핑한 `label` 컬럼을 추가. 그 외 라벨 컬럼(Task E 신규 라벨 등)은 매핑 스킴이 아직 정해지지 않았으므로 raw 값 그대로 반환 — 스킴을 미리 지어내지 않음
- `split_by_ratio()`: 셔플 없이 날짜 순서대로 70/15/15
- `compute_class_weights()`: 빈도 역수 weight tensor
- `NewsDataset`: title+summary를 sentence-pair 토크나이징
- 토크나이저 `AutoTokenizer.from_pretrained('skt/kobert-base-v1')` — sentencepiece 에러 없이 정상(`monologg/kobert` 불필요). 단 XLNetTokenizer 기반이라 `return_token_type_ids=True` 명시 필요, 세그먼트 id가 0/1/2로 나오는데 `type_vocab_size=2`라 `clamp(max=1)`로 방어

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
| `constants.py` | Task T 스트레스 구간 상수 + 날짜 상수(`NEWS_BACKFILL_START/END`, `STOCK_INITIAL_LOAD_START`, `TRAIN_PERIOD_START/END`, `SPLIT_RATIOS`) + **종목 마스터**(`STOCKS`/`STOCK_NAMES`/`ACTIVE_TICKERS`, `namedtuple` 기반, 2026-08-30 중앙화 — 이전엔 `주가_공통.py`/`뉴스_공통.py`/`뉴스_일일수집.py` 세 곳에 흩어져 있었음) | 핵심 |

#### `constants.py` 상수 배경 (2026-08-30, 주석 정리로 코드에서 이관)

코드 쪽 주석은 "어디에 쓰이는지" 한 줄만 남기기로 하고(실행에 영향을 주는 경고만 예외), 근거·
수치·이력 같은 배경 설명은 여기로 옮겼다.

- **`STRESS_PERIOD_START`/`END`**(`"2026-02-02"`~`"2026-07-31"`, Task T 전용): 005930 일별
  |등락률| 10일 이동평균 기준 변곡점 탐지. 2026-01월 1.3~2.0%대에서 2026-02-02(-6.3%)/
  02-03(+11.4%)를 기점으로 3~6%대로 급변, 데이터셋 끝(2026-07-31)까지 지속됨. KOSPI 지수
  월평균 |등락률|로 동일 시점 교차검증 완료(2026-01: 1.15% → 02: 2.49% → … → 07: 4.89%).
  2026-08-17 확정. 탐지 스크립트는 `analysis/detect_stress_period.py`(자동 재탐지 금지 —
  재평가가 필요해지면 사람이 수동 재실행 후 상수를 직접 갱신하고 사유를 커밋 메시지에 기록).
  ⚠️ 이 값은 Task T(가격+거시지표 트랜스포머) 전용이며 D-3(뉴스/감성 트랙, 2026-08-30
  개정으로 전체 기간 사용)과는 무관한 별개 결정이다.
- **`NEWS_BACKFILL_START`/`END`**(`"2023-08-23"`~`"2026-08-23"`): 뉴스 크롤러(`뉴스_최초적재.py`)
  백필 대상 범위. **고정값** — 실행 시점 기준 "오늘로부터 3년 전"으로 자동 재계산되지 않는다.
  재현성 확보 목적. 2026-08-23 3년 유지로 확정(상세는 아래 로드맵 "뉴스 백필 완료" 절).
- **`STOCK_INITIAL_LOAD_START`**(`"2020-01-01"`): 옛 `주가데이터/FinanceData_load.py`의
  하드코딩 값을 2026-08-23 구조 정리 때 이관한 것 — 임의로 고른 값이 아니라 기존 동작을
  그대로 보존한 값이다.
- **`TRAIN_PERIOD_START`/`END`**(`"2023-08-23"`~`"2026-08-28"`, 2026-08-30 신설): 뉴스/감성
  트랙(Task E/F) 학습에 쓸 기간. `NEWS_BACKFILL_START/END`와 **의미가 다르다** — 후자는
  "크롤러가 수집할 범위"(수집/적재 경계, 뉴스 최초적재 스크립트의 기본 인자값)이고 전자는
  "학습에 쓸 범위"(모델링 경계)다. START는 두 값이 같다(뉴스 커버리지가 시작점을 제약하므로) —
  END는 다르다: `NEWS_BACKFILL_END`는 최초 백필 목표로 고정된 스냅샷(2026-08-23)이지만, 이후
  일일수집(`뉴스_일일수집.py`)·주가/시장지표 일일수집이 계속 데이터를 쌓아 실제 커버리지는 더
  늘어났다. `TRAIN_PERIOD_END`는 2026-08-30 시점에 `daily_stock_prices`/`daily_news`
  (`source='search_backfill'`)/`market_indicators`(`KOSPI`) 세 핵심 소스가 공통으로 커버하는
  최신 날짜(2026-08-28)를 DB 실측으로 확인해 넣은 값이다. ⚠️ `NEWS_BACKFILL_START/END`와
  마찬가지로 **고정값**이며 데이터가 더 쌓여도 자동으로 늘어나지 않는다 — Task E 착수 전
  재검증(DB 실측) 후 필요하면 사람이 갱신할 것. ECOS 월간 지표(선행지수순환변동치·M2 등)는
  갱신 주기가 더 성겨(예: 선행지수 최신 기준월 2026-06) `feature_loader.py`의
  `published_date < 거래일` 조건으로 이미 누수 없이 처리되므로 이 상수 자체를 월간 지표
  최신치에 맞출 필요는 없다.
- **`SPLIT_RATIOS`**(`(0.70, 0.15, 0.15)`, 2026-08-30 신설): D-8의 70/15/15 시계열 분할 비율을
  상수로 중앙화 — 종목 마스터를 중앙화한 것과 같은 취지로, 스크립트마다 숫자를 따로 적어 값이
  갈라지는 것을 방지한다.

### 트랙 1 — 주가/거시 (Task G, 전체 완료 ✅)

2026-08-23 구조 정리로 초기적재/일일수집이 파일 단위로 분리됨(같은 날 재작업: 처음엔 초기적재가
일일수집 함수를 직접 import하는 종속 구조였다가, 뉴스 트랙의 `뉴스_공통.py` 패턴에 맞춰
`주가_공통.py`/`지표_공통.py`를 두 파일이 대등하게 참조하는 구조로 다시 정리함 — 초기적재와
일일수집은 서로를 import하지 않는다). 날짜 상수는 `constants.py`(`STOCK_INITIAL_LOAD_START`)로
중앙화 — 시장지표 로더는 "고정 시작일" 개념 자체가 없어(전체 기간 조회가 기본 동작) 상수화
대상에서 제외했다(사람 확인 완료).

| 순서 | 파일 | 역할 | 의존성 |
|---|---|---|---|
| 1 | `주가데이터/주가_공통.py` | `TICKERS`(2026-08-30부터 `constants.ACTIVE_TICKERS` 참조), `update_stock_data()`(핵심 fetch+upsert) 등 공용 로직 | `constants`, `db_manager` |
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

### 트랙 2 — Transformer 가격예측 (Task T-1 완료 ✅ → 방향예측 폐기, 2026-09-06부터 변동성 예측으로 전환, 2026-09-07 100종목 검증으로 최종 확정)

⚠️ **2026-09-06 갱신**: 아래 Task T-1(방향예측) 서술은 역사적 기록으로 그대로 둔다. 이후
방향예측을 8종목 pooled+종목임베딩으로 재시도했으나 동일 결론(무작위 수준)이 재확인돼
**방향예측 트랙 전체를 폐기**했다(`가격예측_공통.py`/`가격예측_일일수집.py`/
`test_evaluation_방향예측.py`는 `쓰래기통/`으로 이동, 상세는 `결과_TaskT_방향예측_폐기.md`).
`가격예측_통합모델.py`(pooled 방향예측)는 폐기 확정이지만 오늘 만든 변동성 실험 스크립트
대부분이 여기서 하이퍼파라미터 상수(BATCH_SIZE/D_MODEL 등)를 가져다 써서 **삭제 불가 —
보존**(상수를 별도 모듈로 분리하기 전까지).

**최종 확정(변동성 예측 트랙, 2026-09-07)**: GARCH(1,1)/하이브리드 Transformer/Parkinson
변동성 추정량/원본시퀀스 실험까지 8종목으로 진행했을 때는 Parkinson-SMA20만 GARCH·SMA20을
이겼고 학습 기반 모델은 못 넘었으나, 종목을 8→50→**100개**로 확장하며 pooled 하이브리드
Transformer가 GARCH·SMA20·Parkinson-SMA20 **세 baseline 전부**를 원본·종목평균 제거
(demean) 지표 양쪽에서 통계적으로 유의하게 이기는 것으로 최종 확정됐다(100종목 N=5 셔플
격차 90~120σ, 50종목 N=30 셔플 p=0.0323 — 상세는 `결과_TaskT_변동성예측_최종.md`
[10]~[15]). demean 분해 검증으로 "분산 확대가 종목 간 평균 차이일 뿐"이라는 우려도
기각됐다. **다만 demean r=0.2795(설명력 R²≈7.8%)로 절대적 설명력은 약하고, 극단치(급변일)
예측이 여전히 약하며, walk-forward 재검증과 실제 의사결정 백테스트가 없어 "통계적으로
유의한 개선"이지 "실전 투입 가능한 수준"은 아직 아니다** — 상세는 결과 문서의 "최종 상태"
절 참고.

100종목 검증 결과를 그대로 옮긴 자동화 파이프라인도 작성 완료했다(코드만, 실행/스케줄러
등록은 안 함): `가격예측/가격예측_변동성_공통.py`(pooled 학습+GARCH 파라미터 캐싱(월 1회
재추정)+baseline 3종 계산+배포 게이트+다음 거래일 예측 저장 — 전체 로직), `가격예측/
가격예측_변동성_일일수집.py`(CLI 진입점), `가격예측/종목_월간갱신.py`(코스피 시가총액
재조회 → 신규 진입 종목은 가격 자동 백필+train 시퀀스 수(≥10) 게이팅 후 활성화, 이탈
종목은 `active=False` — `constants.py`를 통째로 재생성하지 않고 해당 줄만 정밀 수정,
기본 `--dry-run`). 배포 게이트(`train_common.passes_deployment_gate_volatility`)는
GARCH·SMA20·Parkinson-SMA20 셋 다 이겨야 통과로 확정(인자 확장). `model_predictions`
테이블에 `parkinson_sma20_baseline`/`gate_vs_parkinson` 컬럼 추가(SQL 파일 갱신, 운영
DB 적용은 미실행).

신규/확장 핵심 모듈(2026-09-06~07 누적): `가격예측/garch_baseline.py`(GARCH baseline+
leak-free 재귀), `가격예측/pooled_dataset.py`(다종목 pooled 시퀀스 빌더, 8→50→100종목
확장에도 코드 수정 없이 대응), `가격예측/parkinson_volatility_check.py`·`가격예측/
raw_sequence_check_and_train.py`(서로 참조, 재사용 함수 포함이라 보존), `가격예측/
test_evaluation_pooled50_hybrid.py`·`가격예측/test_evaluation_pooled50_hybrid_shuffle30.py`
(50종목 재학습+N=30 셔플 검증), `가격예측/test_evaluation_pooled50_decompose.py`(종목평균
제거 분해 검증), `가격예측/test_evaluation_pooled100_hybrid.py`(100종목 재학습+baseline
비교+decompose+N=5 셔플 통합 스크립트). `가격예측/model.py`에 `PooledTransformerRegressor`
추가(기존 `TransformerRegressor`는 그대로 유지). 조기경보 검증 3종 스크립트·
`test_evaluation_pooled_volatility.py`·`overfit_diagnosis.py`·`training_collapse_check.py`는
결과가 문서화된 뒤 삭제 완료(2026-09-06). 전체 경과·수치표는 `결과_TaskT_변동성예측_최종.md`가
정본이며 여기서는 중복 기재하지 않는다.

⚠️ **종목 마스터가 100개로 확장됨(2026-09-06 50개 → 2026-09-07 100개)** — `constants.STOCKS`가
8종목에서 시가총액 상위 100종목(우선주 제외)으로 확장됐고 000660(SK하이닉스)도 활성화됨.
`034730`은 "SK"로 이름 정정(구 "현대건설", 실제 종목 데이터는 처음부터 정확했음). 100종목
확장 시 상위 51~100위 후보 중 신규 상장으로 train 표본이 지나치게 얇은 3종목(0126Z0/064400/
062040)은 제외하고 다음 순위(036570/088980/052690)로 대체했다. 상세는 위 갱신 문단과
`결과_TaskT_변동성예측_최종.md` [13] 참고.

#### Task T-1 (완료 ✅ — "무작위 수준" 확정, 아래는 당시 기록)

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
| `뉴스데이터/뉴스_공통.py` | search.naver.com 크롤링 공용 상수·함수(`crawl_day`, `fetch_search_page`, `passes_quality_filter`, `upsert_articles`, `clean_title`/`clean_press` 등) | **핵심.** `clean_title`/`clean_press`/`upsert_articles`는 `NaverFinanceNews.py`에서 이관 — 살아있는 코드가 폐기 파일에 의존하던 역방향 구조를 바로잡음. 2026-08-29: 예방적 쿨다운 추가 — `fetch_search_page` 누적 호출이 80회에 도달할 때마다 90~120초 대기(`PREVENTIVE_COOLDOWN_EVERY`/`_RANGE`), 기존 회로차단기(`CONSECUTIVE_FAILURE_LIMIT`/`BACKOFF_SECONDS`)와는 별개로 얹은 사전 예방 조치. 모듈 전역 카운터라 `run_backfill_resumable`이 날짜를 넘나들며 호출해도 하루 단위로 끊기지 않고 누적됨. 2026-08-30: `STOCK_NAMES`(종목코드→회사명)를 `constants.STOCKS`에서 파생하도록 변경 — 이전엔 이 파일에 로컬 하드코딩돼 있었음. 의존: `constants`(신규, 종목 마스터). `db_manager`는 여전히 미의존(순수 크롤링/텍스트 유틸) |
| `뉴스데이터/뉴스_최초적재.py` | 명시적 날짜범위(기본 `NEWS_BACKFILL_START`~`END`) 대량 백필, 체크포인트 기반 재개 | **핵심(현재 승인된 유일한 백필 소스, 2026-08-23 전면 백필 승인, 2026-08-30 완료)**. 의존: `constants`, `db_manager`, `뉴스데이터.뉴스_공통` |
| `뉴스데이터/뉴스_일일수집.py` | 종목별 DB 최신 수집일(`source='search_backfill'` 기준)+1 ~ 오늘 캐치업. 공백 14일 초과 시 자동 캐치업 안 하고 경고. `ACTIVE_TICKERS`는 2026-08-30부터 `constants.ACTIVE_TICKERS` 참조(이전엔 이 파일에 로컬 하드코딩) | **핵심(신규, 아직 스케줄러 연결 전 — 로드맵 "이후" 단계에서 자동화 예정)**. 의존: `constants`, `db_manager`, `뉴스데이터.뉴스_공통` |
**삭제 완료(2026-08-23)**: `NaverFinanceNews.py`(finance.naver.com 종목뉴스, `source='finance_crawl'`
— 페이지네이션 약 1주일 한계로 백필 부적합해 폐기, `clean_title`/`clean_press`/`upsert_articles`는
`뉴스_공통.py`로 이관 완료), `NaverSearchBackfill.py`(위 3개 파일로 분리된 원본), `NaverNews.py`
(네이버 뉴스 검색 API로 당일 뉴스만 수집하던 파일, DB 기여 0행이라 폐기), `NaverNewsCrawl.py`
(네이버 뉴스 검색 웹페이지 크롤링으로 변동성 임계값 초과일의 D-1·D 이틀치를 수집하던 파일,
발행시각 미수집·UI텍스트 혼입 등 결함으로 뉴스_공통/최초적재/일일수집으로 대체 완료), `sentiment_score_label.py`
(CLAUDE.md D-1이 폐기한 구 5-tier 절대임계값 라벨 체계 그 자체 — **Task E, 초과수익률+롤링표준화
구현 전까지 대체 라벨 소스가 없다는 점에 주의**). 삭제 전 grep으로 다른 파일의 실제 import 의존
없음을 확인했다.

**라벨(Task E, `daily_labels` 적재/보고)**:

| 파일 | 역할 | 상태 |
|---|---|---|
| `라벨/라벨_공통.py` | z-score 계산(`compute_z_scores` h=1 전용, `compute_multi_horizon_z_scores` h=1/3/5/10 공용) + 라벨 파생 함수(`label_direction_5class` 등, 조회 시점 전용) + `daily_labels` 조회/split 유틸(`load_labels`, `split_dates_by_ratio`) | **핵심.** 라벨_생성.py/라벨_보고.py/감성분석의 taskf_*.py가 공유. `compute_z_scores`는 라벨_보고.py의 레짐 특성 계산(원 change_rate 필요)이 계속 참조해 그대로 남겨둠 — 신규 코드는 `compute_multi_horizon_z_scores` 사용. 의존: `constants`, `db_manager`, `시장지표.feature_loader` |
| `라벨/라벨_생성.py` | `daily_labels` 적재 — 윈도우 20/60/120 x horizon 1/3/5/10 계산 후 UPSERT | 실행 완료(2026-09-02, 19,608행, 기존 h=1 4,902행은 값 변경 없이 보존). 재실행 안전(멱등, IS DISTINCT FROM). 의존: `constants`, `db_manager`, `라벨.라벨_공통` |
| `라벨/라벨_보고.py` | Task E 정지 지점 보고(z 분포/라벨 분포/split별 비교/D-8 레짐 특성) — h=1(익일) 전용 | 실행 완료(2026-08-31). 상세 결과는 `결과_TaskE_라벨링.md`. 의존: `constants`, `db_manager`, `라벨.라벨_공통` |

| `감성분석/kobert_dataset.py` | `daily_news`/`daily_news_bigkinds` 공용 로더(`load_labeled_news`)+split+Dataset | 2026-08-30 확장: `table_name`/`label_column` 파라미터 추가로 두 테이블을 같은 파이프라인으로 로드 가능(상세는 아래 "각 스크립트 상세 동작" 참고). 2026-09-01 `source` 필터 파라미터 추가(`daily_news.source` 값으로 필터, Task F가 `search_backfill`만 쓰기 위해 사용). 여전히 **기본값은 `sentiment_score`(폐기 대상 라벨)** — Task E가 신규 라벨 컬럼을 만들기 전까지는 기본 호출로 실질 사용 불가. 의존: `db_manager` |
| `감성분석/baseline_tfidf.py` | TF-IDF+로지스틱회귀 진단 베이스라인 | 실험·진단용(Task A에서 1회 사용). 의존: `감성분석.kobert_dataset` |
| `감성분석/kobert_train.py` | KoBERT 파인튜닝 | ⚠️ **재실행 금지**(데이터 재구축 전까지, CLAUDE.md 기 명시). 의존: `감성분석.kobert_dataset` — import 방식을 `from 감성분석.kobert_dataset import ...`로 수정 완료(2026-08-23, 다른 파일들과 관례 통일). `import 감성분석.kobert_train`으로 ImportError 없음 확인(단 `__main__` 블록은 재학습을 바로 시작하므로 재학습 금지 원칙에 따라 실제 실행으로는 검증 안 함) |
| `감성분석/taskf_gating.py` | Task F 게이팅(h=1, 익일) — 라벨 3종 x 윈도우 3종 = 9개 조합 TF-IDF(char_wb 2-4gram)+로지스틱회귀. 뉴스-라벨 매칭(D보다 뒤인 첫 거래일, `allow_exact_matches=False`) | 실행 완료(2026-09-01, 당일 매칭 누수 발견·수정 반영). 결과만으로는 신호 착시였음이 이후 taskf_validate.py로 드러남 — 상세는 `결과_TaskF_게이팅검증.md`. 의존: `constants`, `감성분석.kobert_dataset`, `라벨.라벨_공통` |
| `감성분석/taskf_validate.py` | taskf_gating.py(h=1) 결과 검증 — 레이블 셔플(30회)/시드-부트스트랩 대체/사전분포 무작위 예측기(200회)/표본크기 | 실행 완료(2026-09-01) — 검증 대상 3개 조합 전부 셔플 분포와 통계적으로 구분 안 됨(신호 미검출). 의존: `constants`, `감성분석.taskf_gating` |
| `감성분석/taskf_gating_horizon.py` | Task F 게이팅+검증 통합(h=3/5/10, 윈도우 N=60 고정) — 블록 셔플(누적 윈도우 상관 보존) + 사전분포 무작위 예측기를 게이팅과 함께 즉시 실행 | 실행 완료(2026-09-02) — 9개 조합 전부 신호 미검출, horizon을 늘려도 개선 경향 없음. 의존: `constants`, `감성분석.kobert_dataset`, `감성분석.taskf_gating`, `라벨.라벨_공통` |
| `감성분석/taskf_gating_basis.py` | Task F basis 비교(`label_basis` excess_return vs absolute_return, h=1/3, 윈도우 N=60 고정) — 블록 셔플 + 사전분포 무작위 예측기 동반 | 실행 완료(2026-09-06) — 12개 조합 전부 신호 미검출, "KOSPI 차감으로 시장 전체 신호가 상쇄된다" 가설 기각. 상세는 `결과_TaskF_게이팅검증.md` [5]. 의존: `constants`, `감성분석.kobert_dataset`, `감성분석.taskf_gating`, `감성분석.taskf_gating_horizon`, `라벨.라벨_공통` |
| `감성분석/taskf_gating_relevance_filter.py` | Task F 키워드 관련성 필터 사후 비교(제목에 사건성 키워드 매칭된 기사만 vs 전체) — 필터는 런타임 정규식일 뿐 원본 미변경. 블록 셔플(N=200) + 사전분포 무작위 예측기 | 실행 완료(2026-09-06) — 8개 조합 전부 신호 미검출. 근접했던 조합은 "파업" 단일 사태 편중(관련기사의 81%)과 셔플 해상도 문제(N=30→p=1/31 경계 오판)가 원인이었음을 ablation으로 확인. 상세는 `결과_TaskF_게이팅검증.md` [6]. 의존: `constants`, `감성분석.taskf_gating`, `감성분석.taskf_gating_basis`, `감성분석.taskf_gating_horizon` |
| `감성분석/finbert_gating.py` | Task F 최고 성능 조합(excess_return/방향3-class/N=60/h=3/전체기사)으로 KR-FinBERT(`snunlp/KR-FinBert`) 실제 파인튜닝 — 부분 freeze, 고정 인코더+헤드 재학습 셔플(N=50) 절충안 + 사전분포 무작위 예측기 | 실행 완료(2026-09-06) — test macro F1(0.3257)이 사전분포 무작위 예측기 기대값(0.3269)보다 낮음, 셔플 검증도 유의하지 않음. "KR-FinBERT 투입 보류" 확정. snunlp/KR-FinBert가 safetensors 미제공이라 스크립트 내에서만 `check_torch_load_is_safe`를 no-op 패치해 로드(다른 파일 영향 없음). 상세는 `결과_TaskF_게이팅검증.md` [7]. 의존: `constants`, `감성분석.taskf_gating`, `감성분석.taskf_gating_basis`, `감성분석.taskf_gating_horizon`, `라벨.라벨_공통`, `transformers`, `torch` |

**QA/검증용 별도 도구 (일회성, 파이프라인 아님)**: `recrawl_suspect_dates.py`(53일)/
`recrawl_cluster_expand.py`(91일)/`recrawl_gap_mar_jun.py`(47일, 부분)로 3차에 걸쳐 진행됐다.
결과가 로드맵의 "뉴스 백필 완료 + QA 결과" 절에 전부 기록된 뒤 세 스크립트 모두 삭제됨
(2026-08-30) — 연속 날짜 범위 재크롤링에 별도 스크립트를 새로 만들지 않는다는 원칙도 그 절에
함께 기록돼 있다.

### 트랙 3 부가 — BigKinds 데이터 소스 (2026-08-29 신규, Task A~F 체계 미편입)

`daily_news`(search_backfill 계열)와는 완전히 분리된 별도 실험 소스. Task A~F 어느 단계에도 아직
공식 편입되지 않았고, 기존 뉴스/감성 트랙과 어떻게 관계지을지(대체/보완/별도 비교)는 사람이 아직
결정하지 않았다.

| 파일 | 역할 | 상태 |
|---|---|---|
| `빅카인즈/빅카인즈_적재.py` | `빅카인즈_뉴스/`(gitignore됨) 폴더의 BigKinds CSV/Excel 일괄 적재 → `daily_news_bigkinds` | 최초 적재 완료(20,053행, 2026-08-29). `뉴스 식별자` dtype 버그 수정 반영됨(위 DB 스키마 절 참고). 의존: `db_manager` |

옛 `쓰래기통/Bigkinds.py`(daily_news 대상, CSV 전용, 레거시)는 참고용으로 복원했다가 재사용하지
않고 새로 작성했으며, 이후 삭제함(2026-08-30).

### 감성 통합 실험의 스코프 (2026-08-23 명시)

**감성 피처 통합 실험은 뉴스 커버리지(`constants.TRAIN_PERIOD_START`~`TRAIN_PERIOD_END`,
2023-08-23~2026-08-28, 약 3년 — 2026-08-30부터 이 상수로 관리, 이전엔 `NEWS_BACKFILL_START/END`를
그대로 인용)로 제한되며, 이는 기존 T-1(가격+거시지표 전용, 2020~2026 전체) 학습 범위와 다르다 —
서로 다른 서브 실험으로 명확히 구분한다.** 향후 "가격+거시 vs 가격+거시+뉴스"
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

### 뉴스/감성 트랙 (Task A~F) — Task B 완료(백필+QA), Task E 착수 대기

| 단계 | 내용 | 상태 |
|---|---|---|
| A | 원인 확정 진단 | ✅ 완료, 판정 승인 |
| B | 수집기 교체 → search.naver.com 날짜범위 크롤러(`뉴스_최초적재.py`/`뉴스_일일수집.py`/`뉴스_공통.py`)로 확정(2026-08-23, finance.naver.com은 페이지네이션 한계로 폐기). 발행 "시각"은 이 소스로 확보 불가 — Task D가 안 B(완화)로 대체 확정됨 | ✅ **전면 백필 완료 + QA 검증·클러스터 확장 재크롤링·갭 구간 표본 재크롤링 전부 완료** (아래 참고, 최종 14,392행) |
| C | 전면 재수집. 종목은 당분간 005930 단일(2026-08-23 사람 결정, SK하이닉스 드랍). 섹터 분산 확장은 이후 재검토 | 진행 중(B의 백필과 사실상 통합) |
| D | 이벤트 윈도우 재정렬 — **안 B(완화): D-1 09:00~D 09:00 24시간 윈도우로 확정**(2026-08-23, search_backfill이 시각 정보 없음) | ✅ **실제 구현 완료(2026-09-01)** — `daily_news.target_date` 컬럼을 채우는 방식이 아니라 Task F 쿼리 시점에 `merge_asof`로 즉석 조인하는 방식으로 구현됨. 상세는 아래 "Task F 게이팅·검증·horizon 확장 실험" 참고 |
| E | 라벨 생성 — D-2 롤링 z-score, 방향(5/3-class)·변동성(2-class) 라벨. `daily_labels`에 z_score만 저장(2026-08-30 설계 확정, 위 DB 스키마 절 참고). 2026-09-02 `horizon_h` 컬럼 추가로 익일(h=1) 외 h=3/5/10 누적 라벨도 지원. 상세는 `TASK_EF_라벨링_비교실험.md` 참고 | ✅ **완료** — 임계값·윈도우 확정(2026-08-31) + horizon 확장(2026-09-02). 실행 결과·상세 표는 `결과_TaskE_라벨링.md` 참고 |
| F | 비교 실험 — KR-FinBERT(KoBERT 아님), 데이터소스 교차평가(크롤링/빅카인즈) 포함. 상세는 `TASK_EF_라벨링_비교실험.md` 참고 | ✅ **게이팅·검증·KR-FinBERT 실제 검증까지 완료(2026-09-01~09-06)** — 익일(h=1)·주간(h=3/5/10) 라벨, label_basis(초과/절대수익률), 키워드 관련성 필터 전부 셔플/사전분포 검증에서 신호 미검출. **KR-FinBERT를 최고 성능 조합으로 실제 파인튜닝한 결과도 사전분포 무작위 예측기보다 낮은 성능으로 확인돼 "투입 근거 없음"이 확정됐다.** 상세는 아래 및 `결과_TaskF_게이팅검증.md` 참고 |

#### 뉴스 백필 완료 + QA 결과 (2026-08-30, 클러스터 확장 재크롤링·갭 구간 표본 재크롤링까지 전부 완료)

**전면 백필 완료.** `python -m 뉴스데이터.뉴스_최초적재 --tickers 005930 --start 2023-08-23 --end 2026-08-23`
실행 완료, `source='search_backfill'` 기준 2023-08-23~2026-08-28 전 기간 수집됨(체크포인트
`뉴스데이터/checkpoints/search_backfill_progress.csv`, git 미추적). QA 시점 14,077행 → 1차
재크롤링(53일) 후 14,219행 → 2차 클러스터 확장 재크롤링(91일) 후 14,350행 → 3차 갭 구간 표본
재크롤링(47일) 후 **최종 14,392행**(총 +315건).

**날짜별 수집 건수 분포** (수집 기록 있는 날짜 1,022일 기준): 평균 13.77 / **중앙값 11** / Q1 7 /
Q3 17 / 최대 92. ⚠️ **정정**: 이전에 "2026-07/2023-09 중앙값 15~18건"으로 적었던 참고치가
부정확했다 — 실측 결과 **2026-07은 17건**이 맞지만 **2023-09는 11건**이다.

**언론사 필터 정상 작동 확인**: `source='search_backfill'` 고유 언론사 23개 = `PRESS_WHITELIST`
23개와 정확히 일치, 화이트리스트 밖 press 값 0건.

**빅카인즈(`daily_news_bigkinds`)와의 일별 건수 상관계수**: 0.729 (같은 기간, 언론사 화이트리스트가
서로 달라 완전히 일치하지는 않음 — 해석 시 감안).

**1차 — 의심 날짜 재크롤링 (완료)**: ±7일 롤링 중앙값 대비 50% 이하로 낮았던 거래일 53개를
`analysis/recrawl_suspect_dates.py`로 재수집(별도 체크포인트, `search_backfill_progress.csv`는
미변경). **11개 날짜에서 +142건 신규 발견** — 그중 **2024-07-09~2024-08-01 클러스터 6일에서
+116건(전체 신규의 82%)이 집중**됐다(2024-02-07/02-16도 +21건으로 유사 패턴).

**2차 — 클러스터 확장 재크롤링 (완료)**: 위 두 클러스터의 인접 날짜까지 확장한
2024-02-01~02-29 + 2024-07-01~08-31(총 91일)을 `analysis/recrawl_cluster_expand.py`로 재검증.
**+131건 추가 발견**, `blocked`/`degraded` 0건(`page_errors`도 2024-07-05 1건 제외 전부 0).

| 구간 | 일수 | 신규 발견일 | 신규 건수 |
|---|---|---|---|
| 2024-02 | 29 | 10일 (34%) | 31건 |
| 2024-07 | 31 | 11일 (35%) | 84건 |
| 2024-08 | 31 | 11일 (35%) | 16건 |

가장 큰 발견은 2024-07-31(+19), 2024-07-14(+19), 2024-07-08(+16) — **6일(각 ≥6건)이 131건 중
78건(60%)을 차지하며 전부 7월**. 8월은 잔여치(16건, 대부분 +1~2)만 남아 8월 들어 문제가 빠르게
해소됐음을 시사한다.

⚠️ **확정 사항(가설 아님)**: 두 재크롤링 모두 `page_errors=0`(2024-07-05 1건 제외)이었는데도
대량 누락이 있었다. 즉 **403이 명시적으로 뜨지 않으면서 응답 내용만 비정상인 경우가 실재**하며,
현재 `degraded` 판정 로직(`page_errors`가 반드시 함께 있어야 트리거)으로는 이런 사각지대를 잡지
못한다는 것이 실측으로 확인됐다 — `뉴스데이터/뉴스_공통.py`의 `crawl_day()` 주석에 이미 가설로
적혀 있던 패턴("완전 차단은 아니지만 요청 빈도 제한이 걸린 상태의 응답이 구조는 파싱되지만
내용이 정상이 아니었던 것")의 실증 사례다.

**3차 — 갭 구간(2024-03~06) 표본 재크롤링 (부분 확인 후 중단, 2026-08-30)**: 2024-02와
2024-07~08 두 클러스터 사이 미확인 구간이 "지속된 문제"인지 "독립된 두 사건"인지 판단하기 위해
2024-03-01~2024-06-30(총 122일 계획)을 재검증 시작. **2024-03-01~2024-04-16(47일, 계획의
약 39%) 완료 시점에서 중단** — `blocked`/`degraded` 0건, 신규 21일/47일, **신규 42건**
(일평균 0.89건). 상위: 2024-03-13(+12), 2024-03-12(+6), 2024-03-27(+3) — 12~13일 이틀에
18건(전체의 43%)이 몰린 것을 빼면 나머지는 대부분 +1~2건의 잔여치 수준. 클러스터 구간(2024-02
평균 1.07건/일, 2024-07 평균 2.71건/일)과 비교해 **뚜렷하게 낮은 강도**로 판단해, 사람 판단으로
나머지 75일 재검증의 실익이 낮다고 보고 중단했다(2026-08-30). 사용 스크립트는
`analysis/recrawl_gap_mar_jun.py`였으며, 결과 반영 후 삭제됨(아래 "재크롤링 스크립트 정리" 참고).

**범위 판정 — 2024-02/2024-07~08은 조용한 누락이 집중된 구간으로 확정, 3년 전체로
일반화하지 않음**: 1차 재크롤링(53일)은 이미 2023-08~2026-08 전 기간에서 통계적 이상치로 뽑힌
표본이었는데, 이 두 클러스터 밖에서는 유의한 신규 발견이 없었다(2026-05-14 +1건 제외). 클러스터
내부에서도 7월(84건) → 8월(16건)로 강도가 빠르게 감쇠하는 패턴, 요일 편중 없음(월~일 고르게
분포)을 종합하면 **2024년 2월과 특히 7월 초~중순에 일시적으로 존재했다가 8월에 해소된 국지적
문제**로 판단한다. 3차 갭 구간 표본(2024-03-01~04-16, 47일)에서는 발견 강도가 클러스터 구간의
약 1/3 수준으로 현저히 낮게 나와, **두 클러스터 사이 기간에 같은 수준의 문제가 지속됐을 가능성은
낮다**고 판단한다. 다만 이는 3~6월 전 구간(122일) 완전 검증이 아니라 앞쪽 47일 표본 기준이다 —
재검증 커버리지는 총 **144일(1·2차) + 47일(3차) = 191일 / 1,097일(약 17%)**. "다른 곳에 전혀
없다"는 증명은 아니며, 필요하면 미확인 잔여 구간(2024-04-17~06-30)을 다음 검증 후보로 고려할 것.

재크롤링 후에도 건수가 낮은 11개 날짜(≤2건, 1차 재크롤링 기준)는 **원시 후보(candidates_total)
51~95건이 정상 수집됐으나 필터 통과만 적었던 경우**로, 크롤링 실패가 아니라 **실제 저조일로
판정**한다. `2026-07-29`(2026년 코스피 폭락 다음날)가 여기 포함되는데, 이는 "그 폭락이 뉴스가
아니라 레버리지 ETF 강제청산 등 수급 요인이었다"는 기존 분석(위 "2026년 시장 레짐" 절)과
정합한다.

⚠️ **데이터 규모 정정**: 학습에 쓸 실제 크롤링 데이터는 `daily_news` 전체가 아니라
`source='search_backfill'`만이다(최종 14,392건). "daily_news 25,895건"은 legacy(10,618)+
finance_crawl(1,200)+search_backfill(14,077, 재크롤링 전) 합계이며 학습 정본 규모가 아니다.
상세는 D-10 참고.

**재크롤링 스크립트 정리 원칙(2026-08-30)**: `recrawl_suspect_dates.py`/`recrawl_cluster_expand.py`/
`recrawl_gap_mar_jun.py`는 결과가 이 절에 기록된 뒤 모두 삭제됐다. **연속된 날짜 범위 재크롤링은
새 스크립트를 만들지 말고 기존 `뉴스_최초적재.py`를 `--start`/`--end`/`--checkpoint` 인자로
실행할 것** — 별도 스크립트가 정당한 경우는 "여러 곳에 흩어진 날짜 목록"처럼 연속 범위로 표현
불가능한 대상일 때뿐이다.

**Task B 재수집 완료 후 확인 필수 (기존, Task B 착수 전 조건에서 이관)**: 네이버 금융 종목뉴스
페이지의 과거 조회 가능 기간은 이미 실측 완료(약 4~7일, 부적합 확정) — 이 조건은 해소됨.

#### Task F 게이팅·검증·horizon 확장 실험 (2026-09-01~09-02, 상세는 `결과_TaskF_게이팅검증.md`)

**요약**: 게이팅(TF-IDF+로지스틱회귀)을 익일(h=1) 라벨로 먼저 실행했고, 실행 직후 뉴스-라벨
매칭에서 당일 매칭 누수를 발견해 수정했다. 수정 후에도 9개 조합 전부 baseline을 유의하게
상회해 "보였으나", **셔플 테스트·클래스 사전분포 무작위 예측기로 재검증한 결과 그 "우위"가
전부 평가 방식의 착시였음이 드러났다.** "익일 예측이라 신호가 약했을 수 있다"는 가설로
h=3/5/10(주간 단위 누적 초과수익률)까지 확장해 재실험했지만 결론은 동일했다. **KR-FinBERT
투입은 근거가 없어 보류한다.**

**① 뉴스 라벨 매칭 누수 (2026-09-01)**: `pd.merge_asof(direction="forward")`의 기본값
`allow_exact_matches=True` 때문에 거래일에 발행된 뉴스가 **그날 자신의 라벨**에 매칭되고
있었다 — 시각 정보가 없어 장중 발행인지 장 마감 후 사후 서술인지 구분할 수 없으므로 명백한
누수다. `allow_exact_matches=False`로 "D보다 엄격히 뒤인 첫 거래일"에만 매칭하도록 수정
(Task D "안 B"의 실제 구현 — `daily_news.target_date` 컬럼을 채우는 방식이 아니라 조회
시점에 즉석 조인하는 방식, `TASK_EF_라벨링_비교실험.md` "뉴스 라벨 매칭 규칙" 참고). 수정으로
9개 조합의 baseline 대비 격차(Δtest)가 4~41% 줄었다 — **시점 정렬 없이는 성능이 최대 41%
과대평가될 수 있다**는 것이 실측 수치로 확인됐다.

**② 셔플·사전분포 검증 (2026-09-01)**: 누수 수정 후 남은 "9개 전부 baseline 상회"가 진짜
신호인지 (a) train 라벨만 무작위로 섞고 재학습(30회) (b) 클래스 사전분포를 따르는 무작위
예측기(200회)와 비교했다. 결과: 실제 Δtest가 셔플 분포 95% 구간 **안**에 있었고(3개 조합 중
2개는 셔플 평균보다도 낮음), 실제 macro F1이 사전분포 무작위 예측기의 기대값과 거의 같거나
낮았다. **다수결 baseline을 상회하는 것은 신호의 증거가 아니다** — `class_weight='balanced'`
+ 고차원 희소 TF-IDF 조합은 라벨이 무작위여도 macro F1이 다수결 baseline을 기계적으로
상회하는 경향이 있다(균형 잡힌 예측을 보상하는 지표 특성 + 고차원 노이즈 적합). **⚠️ 앞으로 이
파이프라인 계열로 신호 유무를 판정할 때는 반드시 클래스 사전분포 무작위 예측기와 비교할 것**
— 다수결 baseline 비교만으로는 착시에 빠질 수 있다.

(부수 발견: `LogisticRegression(solver='lbfgs')`는 결정론적이라 `random_state`만 바꾸는 시드
안정성 점검은 무의미했다 — train 부트스트랩 재추출로 대체했고, 결과는 안정적이었다. 즉
"안정적으로 재현되는 착시"였다는 뜻이다.)

**③ Horizon 확장 실험 (2026-09-02)**: "익일 예측이라 노이즈 대비 신호가 약했을 수 있다"는
가설로, 라벨을 h거래일(3/5/10) 누적 초과수익률로 재설계했다(`daily_labels`에 `horizon_h`
컬럼 추가, PK를 `(ticker, date, window_n, horizon_h)`로 확장 — 기존 h=1 데이터는 값 변경
없이 보존, 위 DB 스키마 절 참고). 윈도우 N=60 고정, 라벨 3종 x horizon 3종 = 9개 조합으로
게이팅+검증(**블록 셔플** — 겹치는 누적 윈도우 때문에 생기는 인접 표본 상관을 보존한 채
텍스트-라벨 대응만 끊음)을 함께 실행했다. **결론은 동일했다**(p=0.387~0.903, 전부 유의하지
않음, horizon을 늘려도 신호 대 노이즈 비가 개선되는 경향 없음) — **"익일이라 안 됐다"는
가설은 기각됐다.**

**④ 라벨 basis·키워드 필터 비교, KR-FinBERT 실제 검증 (2026-09-06)**: ①~③의 결론이 라벨
정의나 뉴스 필터링 방식의 한계 때문은 아닌지 추가로 확인했다. (a) KOSPI를 차감하지 않은
절대수익률(`label_basis='absolute_return'`) 라벨로 바꿔도 신호는 나오지 않았다 — "시장 전체
호재가 KOSPI 차감으로 상쇄된다"는 가설 기각. (b) 주가 민감 키워드(계약/수주/소송/인수 등)로
제목을 사후 필터링해 노이즈를 줄여봐도 신호는 나오지 않았다 — 근접했던 조합 하나는 셔플
해상도 문제(N=30→p=1/31 경계 오판, N=200 재검증으로 뒤집힘)와 "파업" 단일 사태 편중이
원인이었다. (c) 가장 성능이 나았던 조합(excess_return/방향3-class/N=60/h=3)으로 **KR-FinBERT를
실제로 파인튜닝**해 TF-IDF와 나란히 비교한 결과, macro F1이 TF-IDF와 사실상 동일했고
**클래스 사전분포 무작위 예측기 기대값보다도 낮았다** — 게이팅 단계의 "투입 보류" 판단이
실제 파인튜닝으로 재확인·확정됐다. 상세는 `결과_TaskF_게이팅검증.md` [5]~[7] 참고.

**종합 결론**: 현재 데이터(005930 단일 종목, 약 3년, test 111거래일) 범위에서는 뉴스 텍스트와
초과수익률 사이에 이 파이프라인으로 검출 가능한 관계가 없다. Task T-1이 "가격+거시지표만으로는
무작위 수준과 구분 안 됨"으로 확정했던 것과 같은 성격의 결론이 뉴스/감성 트랙에서도 확인됐다.
**KR-FinBERT 투입 근거 없음 — 실제 파인튜닝 검증까지 마치고 보류 확정.** 다음 방향 후보(종목/
기간 확장, 데이터소스 교차검증, 문제 재정의 등)는 `결과_TaskF_게이팅검증.md` [8]에 선택지로만
정리했다 — 결정은 아직 하지 않았다.

### 이후
스케줄러/크론 자동화, 추론(서빙) 코드, 컨테이너화(Docker → Kubernetes, 최후순위)

---

## Task C 종목 확장 원칙 (섹터 분산)

현재 실질 분석 대상은 삼성전자 1종목. SK하이닉스(000660)는 확장성 검증용 자리표시자로 등록만 되어 있고 뉴스 0행.

선행연구(정지선·김동성·김종우, 2015, 지능정보연구 21권 4호)에서 **IT 섹터의 뉴스 기반 예측 정확도가 전 섹터 중 최저**였다. IT만으로 구성하면 성능 저하 시 "종목 선정 문제"와 "방법론 문제"가 구분되지 않는다. 섹터를 섞으면 섹터별 성능 비교가 독립적인 분석 축이 된다.

후보: 005930 삼성전자 / 005380 현대차 / 051910 LG화학 / 105560 KB금융 / 207940 삼성바이오로직스 / 035420 NAVER
(000660은 코드 유지, **비활성** — 수집 대상 제외)

**2026-08-30**: 위 종목 목록(코드/회사명/활성 여부)이 `constants.py`의 `STOCKS`(namedtuple 기반)로
중앙화됐다. 이전엔 `주가데이터/주가_공통.py`의 `TICKERS`, `뉴스데이터/뉴스_공통.py`의
`STOCK_NAMES`, `뉴스데이터/뉴스_일일수집.py`의 `ACTIVE_TICKERS`가 각자 따로 관리되고 있었다.
위 후보 5종목은 모두 `active=False`로 이미 등록돼 있어, 실제 확장 시엔 해당 종목의 `active`만
`True`로 바꾸면 된다.

⚠️ **2026-09-06 정정, 2026-09-07 갱신**: `constants.STOCKS`가 가격/변동성 트랙(Task T)
필요에 따라 시가총액 상위 종목으로 확장되면서(2026-09-06 50종목 → 2026-09-07 100종목,
이후 `종목_월간갱신.py`가 월 1회 자동 갱신 예정) 이 절의 "후보 5종목 + 000660 비활성"이라는
전제가 깨졌다 — 현재 100종목 전부 `active=True`다. **단, 이는 가격 데이터(daily_stock_prices)
수집 대상일 뿐 뉴스 수집(Task B/C) 대상이 자동으로 늘어난 게 아니다** — 뉴스 백필은 여전히
005930 단일 종목만 완료된 상태(`daily_news`)이며, `active=True`가 이 절이 원래 의도했던
"뉴스도 함께 수집할 종목"이라는 의미와 이제 일치하지 않는다. 뉴스 트랙에서 종목을
실제로 확장하려면 이 절의 섹터 분산 원칙에 따라 별도로 판단하고 `뉴스데이터/뉴스_최초적재.py`를
그 종목들에 대해 명시적으로 실행해야 한다 — `active=True`만으로 뉴스가 저절로 쌓이지 않는다.

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
