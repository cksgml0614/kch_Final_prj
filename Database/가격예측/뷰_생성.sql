-- model_predictions 위의 조회용 뷰 2개: model_predictions_trusted(신뢰 조건을 만족한 예측만), model_predictions_realized(예측 + 실측).
-- 배경: Database/결정기록.md

-- model_predictions_trusted — 게이트 통과이고 입력 오염 의심이 아닌 행만. 미통과·의심 행까지 봐야 하면 원본 테이블을 조회한다.
-- ⚠️ input_data_suspect는 지수 결측만 본다 — 장중 잠정값으로 만든 예측도 여기 들어올 수 있다.
CREATE OR REPLACE VIEW model_predictions_trusted AS
SELECT * FROM model_predictions WHERE gate_passed = true AND input_data_suspect = false;

-- model_predictions_realized — 정답 = prediction_date 뒤 그 종목의 첫 거래일 t의 |ln(close_t / close_prev)| x 100, close_prev는 t 직전 행.
-- target_date는 쓰지 않는다(공휴일을 몰라 휴장일일 수 있다). 대시보드.백테스트_재생이 같은 정의를 코드로 옮겨 쓴다.
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
