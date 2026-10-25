-- model_predictions·model_registry 테이블 자체를 삭제한다. ⚠️ 운영 데이터가 사라진다 — 반드시 확인 후 실행.
-- 두 테이블 모두 FK가 없어 순서 무관, CASCADE 불필요. 뷰(뷰_생성.sql)가 model_predictions에 의존하므로 뷰_삭제.sql을 먼저 실행한다.

DROP TABLE IF EXISTS model_predictions;

DROP TABLE IF EXISTS model_registry;
