-- model_predictions 테이블 생성 — 변동성 예측 전용으로 재정의(2026-09-06, Task T 방향예측
-- 폐기 + 변동성 예측 전환). 기존 스키마(predicted_return 기반, 방향 예측용)는 완전히
-- 대체됐다 — 상세 배경은 CLAUDE.md, 결과_TaskT_방향예측_폐기.md, 결과_변동성_조기경보_검증.md
-- 참고.
--
-- garch_baseline/sma20_baseline을 함께 저장하는 이유: 이 프로젝트의 배포 게이트가 "하이브리드
-- Transformer가 GARCH·SMA20 둘 다를 이겨야 통과"로 정의돼 있어(passes_deployment_gate_
-- volatility), 매 예측 시점의 세 값을 나란히 남겨야 사후에 게이트 판정 근거를 감사할 수 있다.
-- is_early_warning은 하이브리드 예측 σ가 직전 60일 예측 σ 평균의 1.5배를 넘는지(조기경보
-- 임계값, 결과_변동성_조기경보_검증.md에서 검증된 규칙)를 그대로 저장한다.

CREATE TABLE IF NOT EXISTS model_predictions (
    ticker                VARCHAR(10)  NOT NULL,
    target_date           DATE         NOT NULL,   -- 예측 대상 거래일 t(미래, 예측 시점엔 아직 미실현)
    prediction_date       DATE         NOT NULL,   -- 학습/예측에 사용한 마지막 확정 거래일(t-1)
    predicted_volatility  NUMERIC      NOT NULL,   -- 하이브리드 Transformer 예측치: |로그수익률x100| 추정값
    garch_baseline        NUMERIC      NOT NULL,   -- 같은 시점 GARCH(1,1) 조건부 σ 예측치(감사/비교용)
    sma20_baseline        NUMERIC      NOT NULL,   -- 같은 시점 SMA20(직전 20일 평균|수익률|) 예측치(감사/비교용)
    actual_volatility     NUMERIC,                 -- target_date 확정 후 채움(추후 백필 작업). NULL이면 아직 미확정
    is_early_warning      BOOLEAN      NOT NULL,   -- 예측σ > 직전60일 예측σ평균의 1.5배(조기경보 임계값)
    model_version         VARCHAR(32)  NOT NULL,   -- train_common.save_checkpoint()가 부여한 버전(YYYYMMDD_HHMMSS)
    gate_passed           BOOLEAN      NOT NULL,   -- passes_deployment_gate_volatility() 종합 판정
    gate_vs_garch         BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < GARCH RMSE (val 기준)
    gate_vs_sma20         BOOLEAN      NOT NULL,   -- 하이브리드 RMSE < SMA20 RMSE (val 기준)
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),

    PRIMARY KEY (ticker, target_date)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_prediction_date ON model_predictions (prediction_date);
