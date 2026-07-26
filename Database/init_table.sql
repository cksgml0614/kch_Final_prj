-- 1. 테이블 삭제 (존재할 경우에만 삭제하여 에러 방지)
DROP TABLE IF EXISTS daily_stock_prices CASCADE;
DROP TABLE IF EXISTS daily_news CASCADE;
DROP TABLE IF EXISTS market_indicators CASCADE;

-- (참고) 보통 테이블을 날리면 인덱스도 날아가지만,
-- 혹시 모를 잔여 개체를 위해 스크립트를 깔끔하게 관리하고 싶다면 아래처럼 명시할 수도 있습니다.
-- DROP INDEX IF EXISTS idx_news_date;
-- DROP INDEX IF EXISTS idx_news_ticker;
-- DROP INDEX IF EXISTS idx_indicators_date;