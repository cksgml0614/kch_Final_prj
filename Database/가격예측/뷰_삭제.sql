-- model_predictions_trusted 뷰 삭제 (2026-09-14, D-1/D-3)
-- 뷰만 제거 — model_predictions 원본 테이블/데이터는 건드리지 않는다.

DROP VIEW IF EXISTS model_predictions_trusted;

-- model_predictions_realized 뷰 삭제 (2026-10-11). 이 뷰에 의존하는 객체는 없다.
DROP VIEW IF EXISTS model_predictions_realized;
