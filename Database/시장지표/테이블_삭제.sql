-- market_indicators(자식) → indicator_meta(부모) 순서로 삭제한다. FK 때문에 자식 테이블을
-- 먼저 지워야 부모 테이블을 지울 수 있다(반대 순서로 실행하면 에러).

DROP TABLE IF EXISTS market_indicators;
DROP TABLE IF EXISTS indicator_meta;
