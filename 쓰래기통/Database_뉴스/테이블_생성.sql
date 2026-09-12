-- daily_news 테이블 생성. 2026-08-23 기준 실제 운영 DB 스키마를 information_schema/
-- pg_constraint/pg_indexes로 직접 조회해 작성 — Database/migrations/001, 003이 반영된
-- 최종 상태다(더 이상 마이그레이션 파일을 따로 참고할 필요 없음).

CREATE TABLE IF NOT EXISTS daily_news (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10),
    date            DATE,
    title           TEXT,
    summary         TEXT,
    sentiment_score DOUBLE PRECISION,   -- ⚠️ D-1 폐기 대상 라벨 체계(구 5-tier). 재사용 금지, 비교 기록용으로만 보존
    published_at    TIMESTAMP,          -- migrations/001. search_backfill 소스는 날짜만 채움(00:00), finance_crawl은 실제 시각
    target_date     DATE,               -- migrations/001. Task D 윈도우 재정렬 후 채워짐 (현재 NULL)
    source          VARCHAR(32),        -- migrations/001. 'legacy' / 'finance_crawl' / 'search_backfill'
    press           VARCHAR(64),        -- migrations/001
    article_url     TEXT,               -- migrations/001

    CONSTRAINT uq_news_ticker_date_title_summary UNIQUE (ticker, date, title, summary)
);

CREATE INDEX IF NOT EXISTS idx_news_date ON daily_news (date);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON daily_news (ticker);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON daily_news (published_at);              -- migrations/001
CREATE INDEX IF NOT EXISTS idx_news_target ON daily_news (ticker, target_date);              -- migrations/001

-- migrations/003: 종목 무관 전역 UNIQUE였다가 여러 종목에 공통으로 뜨는 시황 기사가
-- 먼저 처리된 종목에만 저장되는 손실이 발견되어 (ticker, article_url)로 좁힘
CREATE UNIQUE INDEX IF NOT EXISTS uq_news_ticker_url ON daily_news (ticker, article_url) WHERE (article_url IS NOT NULL);
