-- model_predictions 테이블 생성 — 변동성 예측 전용으로 재정의(2026-09-06, Task T 방향예측
-- 폐기 + 변동성 예측 전환). 기존 스키마(predicted_return 기반, 방향 예측용)는 완전히
-- 대체됐다 — 상세 배경은 CLAUDE.md, 결과_TaskT_방향예측_폐기.md, 결과_변동성_조기경보_검증.md
-- 참고.
--
-- garch_baseline/sma20_baseline/parkinson_sma20_baseline을 함께 저장하는 이유: 배포 게이트가
-- "하이브리드 Transformer가 GARCH·SMA20·Parkinson-SMA20 셋 다를 이겨야 통과"로 정의돼 있어
-- (passes_deployment_gate_volatility, 2026-09-07 100종목 검증 후 Parkinson-SMA20 포함 확정),
-- 매 예측 시점의 네 값을 나란히 남겨야 사후에 게이트 판정 근거를 감사할 수 있다.
-- is_early_warning은 하이브리드 예측 σ가 직전 60일 예측 σ 평균의 1.5배를 넘는지를 그대로
-- 저장한다. ⚠️ 이 규칙 자체의 조기경보 "능력"은 결과_변동성_조기경보_검증.md 원 검증이 GARCH
-- 라벨링 버그로 무효화된 뒤 재검증에서 우연 수준으로 확인됐다(CLAUDE.md 참고) — 이 컬럼은
-- 감사/기록용 descriptive 값일 뿐 검증된 운영 신호가 아니다.

CREATE TABLE IF NOT EXISTS model_predictions (
    ticker                    VARCHAR(10)  NOT NULL,
    target_date               DATE         NOT NULL,   -- 예측 대상 거래일 t(미래, 예측 시점엔 아직 미실현)
    prediction_date           DATE         NOT NULL,   -- 학습/예측에 사용한 마지막 확정 거래일(t-1)
    predicted_volatility      NUMERIC      NOT NULL,   -- 하이브리드 Transformer 예측치: |로그수익률x100| 추정값
    garch_baseline            NUMERIC      NOT NULL,   -- 같은 시점 GARCH(1,1) 조건부 σ 예측치(감사/비교용)
    sma20_baseline            NUMERIC      NOT NULL,   -- 같은 시점 SMA20(직전 20일 평균|수익률|) 예측치(감사/비교용)
    parkinson_sma20_baseline  NUMERIC      NOT NULL,   -- 같은 시점 Parkinson-SMA20 예측치(감사/비교용, 2026-09-07 추가)
    actual_volatility         NUMERIC,                 -- target_date 확정 후 채움(추후 백필 작업). NULL이면 아직 미확정
    is_early_warning          BOOLEAN      NOT NULL,   -- 예측σ > 직전60일 예측σ평균의 1.5배(descriptive, 위 주석 참고)
    model_version              VARCHAR(32)  NOT NULL,   -- train_common.save_checkpoint()가 부여한 버전(YYYYMMDD_HHMMSS)
    gate_passed               BOOLEAN      NOT NULL,   -- passes_deployment_gate_volatility() 종합 판정
    gate_vs_garch             BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < GARCH RMSE (val 기준)
    gate_vs_sma20             BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < SMA20 RMSE (val 기준)
    gate_vs_parkinson         BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < Parkinson-SMA20 RMSE (val 기준, 2026-09-07 추가)
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT now(),
    input_data_suspect        BOOLEAN      NOT NULL DEFAULT false,  -- 입력 데이터 오염 의심(2026-09-27 추가, 아래 마이그레이션 참고)

    PRIMARY KEY (ticker, target_date)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_prediction_date ON model_predictions (prediction_date);

-- ── 2026-09-07 마이그레이션 (이미 생성된 운영 DB에 적용할 때) ──────────────────────────
-- 위 CREATE TABLE은 신규 설치 기준이고, 이미 만들어진 테이블에는 아래 ALTER로 반영한다.
-- CLAUDE.md 규칙대로 기존 컬럼은 건드리지 않고 신규 컬럼만 추가(파괴적 변경 없음).
-- ⚠️ 이 세션에서는 실행하지 않았다 — 실제 운영 DB 적용은 가격예측_변동성_일일수집.py를
-- 처음 돌리기 전에 사람이 직접 실행할 것.
--
-- ALTER TABLE model_predictions
--     ADD COLUMN IF NOT EXISTS parkinson_sma20_baseline NUMERIC,
--     ADD COLUMN IF NOT EXISTS gate_vs_parkinson BOOLEAN;
-- -- 컬럼 추가 직후에는 기존 행에 NULL이 들어가므로(과거 행이 있다면), NOT NULL 제약은
-- -- 신규 컬럼에 데이터가 없는 한 걸 수 없다. 이 프로젝트는 아직 model_predictions에 실제
-- -- 적재된 행이 없어(가격예측_변동성_일일수집.py 최초 실행 전) 아래처럼 바로 NOT NULL을
-- -- 걸어도 된다 — 만약 이미 적재된 행이 있다면 백필 후 NOT NULL을 걸 것.
-- ALTER TABLE model_predictions
--     ALTER COLUMN parkinson_sma20_baseline SET NOT NULL,
--     ALTER COLUMN gate_vs_parkinson SET NOT NULL;

-- ── 2026-09-27 마이그레이션: input_data_suspect (운영 DB에 적용 완료) ─────────────────────
-- 배경: FDR 지수 캐시(fdr_krx_data_cache)가 2026-09-17 장중 스냅숏 이후 갱신되지 않아
-- market_indicators의 KOSPI/KOSDAQ가 9/18~ 결측 — 그 상태로 만든 예측(prediction_date
-- 2026-09-18/2026-09-23, 200행)은 KOSPI 레벨 피처가 forward-fill되고 excess_return_z_lag1이
-- KOSPI 수익률 0으로 계산된 오염 입력을 썼다. 소급 재계산하지 않고(기록 보존) 이 플래그로
-- 표시한다. 이후 실행분은 가격예측_변동성_공통.check_market_index_freshness()가 결측을 감지하면
-- 자동으로 true를 기록한다. model_predictions_trusted 뷰가 이 플래그도 걸러낸다(뷰_생성.sql).
-- 컬럼 순서: 운영 DB에서는 ALTER로 맨 뒤(created_at 다음)에 붙었으므로 위 CREATE TABLE도 같은
-- 위치에 두었다.
--
-- ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS input_data_suspect BOOLEAN NOT NULL DEFAULT false;
-- UPDATE model_predictions SET input_data_suspect = true WHERE prediction_date IN ('2026-09-18', '2026-09-23');
-- (같은 날 추가) 9/14 주가 장중 수집 오염(84종목 종가, --force로 정정) — 9/14 입력을 쓴 예측 2행도 표시:
-- UPDATE model_predictions SET input_data_suspect = true WHERE prediction_date = '2026-09-14';

-- 2026-10-11: 모델 버전(체크포인트) 목록. model_predictions.model_version의 계열·출처를 판정한다(CLAUDE.md "MLflow 읽는 법").
-- 최초 적재 54행: 체크포인트 meta 53개(family_basis='checkpoint_meta') + 20260914_132800 1행(체크포인트 없음, 'inferred').
-- source 규칙: n_train < 10000 → 'test', saved_at 2026-09-14 13:39~15:02 UTC이면서 train_end < 2026-09-14 →
-- 'backtest_d4_20260914', 나머지 → 'live'. gate_definition: version < 20260928_085818 → 'pooled', 이후 → 'tickermean'.
-- 새 version 자동 등록은 아직 없다(파이프라인 미연동).
-- FK는 걸지 않는다: 일일 파이프라인이 registry 등록 전에 model_predictions에 쓰므로 FK가 있으면 저장이 실패한다.
CREATE TABLE IF NOT EXISTS model_registry (
    model_version       VARCHAR(32)  PRIMARY KEY,                -- model_predictions.model_version과 같은 형식(UTC)
    model_family        VARCHAR(16)  NOT NULL CHECK (model_family IN ('transformer', 'lstm')),
    model_class         VARCHAR(64),                              -- 체크포인트 meta model_class
    family_basis        VARCHAR(16)  NOT NULL CHECK (family_basis IN ('checkpoint_meta', 'inferred')),
    source              VARCHAR(32)  NOT NULL,                    -- 'live' / 'backtest_d4_20260914' / 'test'
    gate_definition     VARCHAR(16)  NOT NULL CHECK (gate_definition IN ('pooled', 'tickermean')),
    train_end           DATE,                                     -- meta train_end(실행 end, KST 날짜)
    split_train_end     DATE,
    split_val_end       DATE,
    n_train             INTEGER,
    hyperparameter_ref  VARCHAR(64),                              -- 지금은 NULL(월간 튜닝 도입 시)
    note                TEXT,
    registered_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);
