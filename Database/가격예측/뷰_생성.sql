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

CREATE OR REPLACE VIEW model_predictions_trusted AS
SELECT * FROM model_predictions WHERE gate_passed = true;
