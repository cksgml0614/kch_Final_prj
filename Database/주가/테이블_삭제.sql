-- daily_stock_prices 테이블 자체를 삭제한다. 다른 테이블이 이를 FK로 참조하지 않으므로
-- CASCADE 불필요.

DROP TABLE IF EXISTS daily_stock_prices;
