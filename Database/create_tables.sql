-- FinanceDataReader 주가 데이터 테이블
CREATE TABLE IF NOT EXISTS daily_stock_prices (
    ticker VARCHAR(10),     -- 종목코드
    date DATE,              -- 날짜
    open BIGINT,            -- 시작가
    high BIGINT,            -- 상한가
    low BIGINT,             -- 하한가
    close BIGINT,           -- 종가
    volume BIGINT,          -- 거래량
    change_rate FLOAT,      -- 대비율
    PRIMARY KEY (ticker, date)
);
-- 뉴스 데이터 테이블 (BigKinds 및 Naver API 통합용)
CREATE TABLE IF NOT EXISTS daily_news (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),  -- 종목코드
    date DATE,           -- 뉴스 날짜
    title TEXT,          -- 뉴스 제목
    summary TEXT,        -- 뉴스 요약본
    sentiment_score FLOAT DEFAULT NULL, -- 나중에 감성분석으로 채울 예정
    CONSTRAINT uq_news_ticker_date_title_summary UNIQUE (ticker, date, title, summary)
);
-- 날짜별로 뉴스를 검색하는 경우가 많으므로 인덱스 추가 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_news_date ON daily_news(date);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON daily_news(ticker);

-- 모든 종류의 경제/시장 지표를 담는 통합 테이블
CREATE TABLE IF NOT EXISTS market_indicators (
    indicator_code VARCHAR(20), -- 지표 고유 코드 (예: 'US10Y', 'VIX', 'DXY')
    date DATE,                  -- 지표 발생 날짜
    value NUMERIC,              -- 지표 값
    unit VARCHAR(20),           -- 단위 (예: '%', 'index', 'usd')
    PRIMARY KEY (indicator_code, date) -- 동일 지표의 날짜 중복 방지
);

-- 검색 속도 최적화를 위한 인덱스. 날짜만으로 탐색
CREATE INDEX IF NOT EXISTS idx_indicators_date ON market_indicators(date);