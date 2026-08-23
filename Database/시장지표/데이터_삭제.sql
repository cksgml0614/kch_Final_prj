-- market_indicators(자식) → indicator_meta(부모) 순서로 비운다. FK가 ON DELETE RESTRICT라
-- indicator_meta에 참조하는 market_indicators 행이 남아있으면 indicator_meta를 지울 수 없다.
--
-- indicator_meta는 "데이터"라기보다 지표 정의(코드/이름/출처 등) 메타데이터에 가깝다.
-- 지표 값(market_indicators)만 비우고 정의는 남기고 싶다면 아래 두 번째 줄은 빼고 실행할 것.

TRUNCATE TABLE market_indicators;
TRUNCATE TABLE indicator_meta;   -- 지표 정의까지 지우고 싶지 않으면 이 줄을 빼고 실행
