-- daily_news_bigkinds 테이블 생성(2026-08-29). BigKinds CSV 일괄 적재 전용이며 기존 daily_news
-- (search_backfill/legacy/finance_crawl)와는 완전히 별도 테이블 — 서로 FK 관계 없음, 어느 한쪽을
-- 변경해도 다른 쪽에 영향 없음. 빅카인즈/빅카인즈_적재.py가 이 테이블에 적재한다.

CREATE TABLE IF NOT EXISTS daily_news_bigkinds (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10),
    date            DATE,
    title           TEXT,
    summary         TEXT,               -- 본문 전체
    press           VARCHAR(64),
    article_url     TEXT,
    keywords        TEXT,
    category        TEXT,               -- 통합 분류1
    bigkinds_id     TEXT,               -- 뉴스 식별자, 원본 추적/중복방지
    target_date     DATE,               -- Task D 윈도우 재정렬용 (현재 NULL, daily_news와 동일 목적)
    source          VARCHAR(32) DEFAULT 'bigkinds',
    created_at      TIMESTAMP DEFAULT now(),

    CONSTRAINT uq_bigkinds_id UNIQUE (bigkinds_id)
);

CREATE INDEX IF NOT EXISTS idx_news_bigkinds_date ON daily_news_bigkinds (date);
CREATE INDEX IF NOT EXISTS idx_news_bigkinds_ticker ON daily_news_bigkinds (ticker);
CREATE INDEX IF NOT EXISTS idx_news_bigkinds_target ON daily_news_bigkinds (ticker, target_date);
CREATE UNIQUE INDEX IF NOT EXISTS uq_news_bigkinds_ticker_url ON daily_news_bigkinds (ticker, article_url) WHERE (article_url IS NOT NULL);
