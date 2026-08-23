-- daily_stock_prices 테이블 생성. 2026-08-23 기준 실제 운영 DB 스키마 그대로
-- (이 테이블은 생성 이후 마이그레이션 변경 이력 없음).

CREATE TABLE IF NOT EXISTS daily_stock_prices (
    ticker      VARCHAR(10),
    date        DATE,
    open        BIGINT,
    high        BIGINT,
    low         BIGINT,
    close       BIGINT,
    volume      BIGINT,
    change_rate DOUBLE PRECISION,

    PRIMARY KEY (ticker, date)
);
