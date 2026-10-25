-- model_predictions·model_registry 테이블 생성 — 운영 DB의 지금 스키마. model_predictions는 종목·대상일마다 다음 거래일 변동성 예측 1행이고,
-- baseline 3종과 게이트 판정을 함께 남겨 사후에 게이트 근거를 감사할 수 있게 한다.
-- ⚠️ 저장은 (ticker, target_date) UPSERT라 같은 키를 다시 계산하면 이전 값이 남지 않는다. created_at은 최초 INSERT 시각이다.
-- ⚠️ model_predictions에 컬럼을 추가하면 model_predictions_realized 뷰(p.*)를 DROP 후 다시 만든다(뷰_생성.sql).
-- 배경·마이그레이션 이력: Database/결정기록.md

CREATE TABLE IF NOT EXISTS model_predictions (
    ticker                    VARCHAR(10)  NOT NULL,
    target_date               DATE         NOT NULL,   -- 예측 대상 거래일 t(미래, 예측 시점엔 아직 미실현)
    prediction_date           DATE         NOT NULL,   -- 학습/예측에 사용한 마지막 확정 거래일(t-1)
    predicted_volatility      NUMERIC      NOT NULL,   -- 하이브리드 모델(pooled LSTM) 예측치: |로그수익률x100| 추정값
    garch_baseline            NUMERIC      NOT NULL,   -- 같은 시점 GARCH(1,1) 조건부 σ 예측치(감사/비교용)
    sma20_baseline            NUMERIC      NOT NULL,   -- 같은 시점 SMA20(직전 20일 평균|수익률|) 예측치(감사/비교용)
    parkinson_sma20_baseline  NUMERIC      NOT NULL,   -- 같은 시점 Parkinson-SMA20 예측치(감사/비교용)
    actual_volatility         NUMERIC,                 -- 옛 백필 스크립트가 채운 스냅숏(지금은 채우지 않음). 실측은 model_predictions_realized 뷰
    is_early_warning          BOOLEAN      NOT NULL,   -- 예측σ > 직전60일 예측σ평균의 1.5배(기록용 — 검증된 운영 신호 아님)
    model_version              VARCHAR(32)  NOT NULL,   -- store.save_checkpoint()가 부여한 버전(UTC YYYYMMDD_HHMMSS)
    gate_passed               BOOLEAN      NOT NULL,   -- evaluate.passes_deployment_gate_volatility() 종합 판정
    gate_vs_garch             BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < GARCH RMSE (val 기준)
    gate_vs_sma20             BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < SMA20 RMSE (val 기준)
    gate_vs_parkinson         BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < Parkinson-SMA20 RMSE (val 기준)
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT now(),
    input_data_suspect        BOOLEAN      NOT NULL DEFAULT false,  -- 입력 데이터 오염 의심(KOSPI/KOSDAQ 결측 등). 운영 DB와 같은 컬럼 순서라 맨 뒤

    PRIMARY KEY (ticker, target_date)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_prediction_date ON model_predictions (prediction_date);

-- model_registry — 모델 버전(체크포인트)마다 1행. model_predictions.model_version의 계열·출처를 판정한다.
-- 새 version은 store.register_model_version이 체크포인트 저장 직후 등록한다(source 'live', gate_definition 'tickermean').
-- FK는 걸지 않는다 — 등록은 부가 기록이라 실패해도([REGISTRY-SKIP]) 예측 저장은 계속돼야 한다.
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
    hyperparameter_ref  VARCHAR(64),                              -- 쓰지 않는다(항상 NULL). 튜닝값은 MLflow param tuned_params_id
    note                TEXT,
    registered_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);
