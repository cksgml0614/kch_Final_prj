-- model_predictions_trusted 뷰 생성 (2026-09-14, D-1/D-3)
--
-- 배경: A-1(2026-09-12)은 배포 게이트 미통과 시 model_predictions 저장 자체를 스킵하는
-- 엄격한 방식이었으나, 그 결과 게이트가 연속 미통과하는 동안 "미래 예측이 실제로 얼마나
-- 정확한지" 검증할 데이터가 더 이상 쌓이지 않는 문제가 드러났다(actual_volatility 백필로
-- A-1 이전 100행을 실측 대조한 뒤 발견). D-1 결정: model_predictions에는 게이트 통과
-- 여부와 무관하게 항상 저장하고, 기존 gate_passed 컬럼으로 신뢰 가능 여부를 구분한다.
-- 새 테이블은 만들지 않는다.
--
-- 이제 model_predictions에 통과/미통과 예측이 섞이므로, 스크리닝/조회 시 매번
-- `WHERE gate_passed = true`를 기억해야 하는 부담을 줄이기 위한 뷰. 상세 배경은
-- CLAUDE.md "A-1 엄격 스킵 해제" 절 참고.
--
-- ⚠️ gate_passed=false인 행(참고용, 신뢰 불가)을 봐야 하는 감사/분석 목적이라면 이 뷰가
-- 아니라 원본 model_predictions 테이블을 직접 조회할 것 — 이 뷰는 "신뢰 가능한 예측만"
-- 걸러내는 용도로만 쓴다.

-- 2026-09-27: input_data_suspect(입력 데이터 오염 의심) 행도 제외하도록 확장 — gate_passed와 같은
-- 필터링 패턴. 배경은 테이블_생성.sql 하단 2026-09-27 마이그레이션 주석 참고. (운영 DB 적용 완료)

CREATE OR REPLACE VIEW model_predictions_trusted AS
SELECT * FROM model_predictions WHERE gate_passed = true AND input_data_suspect = false;

-- model_predictions_realized 뷰 (2026-10-11)
-- 정답 = prediction_date 뒤 그 종목의 첫 거래일 t의 |ln(close_t / close_prev)| x 100, close_prev는 t 직전 행.
-- target_date는 쓰지 않는다. next_weekday()가 공휴일을 몰라 target_date가 휴장일이면
-- actual_volatility가 영구 NULL로 남던 문제(8/17, 9/24, 10/5, 10/9)를 피한다.
-- 기존 actual_volatility 컬럼과 actual_volatility_백필.py는 그대로 둔다(백필 당시 스냅숏).
-- 가격이 정정되면 이 뷰 값은 자동으로 바뀐다(이력 없음).
-- close <= 0 행은 ln 오류로 쿼리 전체가 실패하지 않도록 NULL 처리한다.
-- ⚠️ p.*는 뷰를 만들 때 컬럼이 고정된다. model_predictions에 컬럼을 추가하면 이 뷰를 DROP 후 다시 만들어야 한다.
CREATE OR REPLACE VIEW model_predictions_realized AS
SELECT p.*,
       nxt.date AS realized_date,
       CASE WHEN nxt.date IS NOT NULL AND prev.close > 0 AND nxt.close > 0
            THEN abs(ln(nxt.close::float8 / prev.close::float8)) * 100 END AS realized_volatility,
       CASE WHEN nxt.date IS NOT NULL THEN 'confirmed' ELSE 'pending' END AS realized_status,
       (nxt.volume = 0) AS target_halted          -- pending이면 NULL
FROM model_predictions p
LEFT JOIN LATERAL (
    SELECT d.date, d.close, d.volume FROM daily_stock_prices d
    WHERE d.ticker = p.ticker AND d.date > p.prediction_date
    ORDER BY d.date LIMIT 1) nxt ON true
LEFT JOIN LATERAL (
    SELECT d.close FROM daily_stock_prices d
    WHERE d.ticker = p.ticker AND d.date < nxt.date
    ORDER BY d.date DESC LIMIT 1) prev ON true;
