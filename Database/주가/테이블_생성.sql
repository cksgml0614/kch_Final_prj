-- daily_stock_prices 테이블 생성 — 운영 DB의 지금 스키마(생성 이후 변경 없음). 종목·거래일마다 OHLCV 1행.
-- 거래정지일 행은 시가·고가·저가 0, 거래량 0, 종가 = 직전 종가로 들어온다.

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
