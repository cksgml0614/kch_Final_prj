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
