-- daily_labels 테이블 생성. Task E(TASK_EF_라벨링_비교실험.md) — 롤링 z-score만 저장하고
-- 라벨(방향 5-class/3-class, 변동성 2-class)은 조회 시 임계값을 적용해 파생한다
-- (2026-08-30 설계 확정, 2026-08-31 구현). 뉴스 테이블(daily_news/daily_news_bigkinds)에는
-- 라벨을 컬럼으로 추가하지 않는다 — CLAUDE.md DB 스키마 절 "daily_labels" 참고.
--
-- 2026-09-02: horizon_h 컬럼 추가(ALTER TABLE로 운영 DB에 적용, 기존 h=1 데이터는 보존).
-- 익일(h=1) 단일거래일 라벨 대신 h거래일 누적 초과수익률 라벨(h=3/5/10)을 같은 테이블에
-- 함께 보관한다 — window_n과 동일하게 PK에 포함시켜 여러 horizon을 한 테이블에서 비교 가능하게
-- 둔다(새 컬럼 추가로 처리, 기존 h=1 행 값은 덮어쓰지 않음).

CREATE TABLE IF NOT EXISTS daily_labels (
    ticker         VARCHAR(10) NOT NULL,
    date           DATE NOT NULL,          -- 거래일 기준(앵커일 t)
    window_n       INTEGER NOT NULL,       -- sigma 계산 윈도우(20/60/120 중 하나)
    horizon_h      INTEGER NOT NULL,       -- 누적 수익률 기간(거래일). 1=익일(기존), 3/5/10=n거래일 누적
    excess_return  NUMERIC,                -- t부터 horizon_h거래일(t..t+h-1) 누적 (종목-KOSPI) change_rate 합. h=1이면 당일 값과 동일. 계산 가능하면 항상 채움
    sigma          NUMERIC,                -- 직전 window_n개 앵커일의 동일 horizon_h 누적값 표준편차(t 시점 미포함). 버퍼 부족 시 NULL
    z_score        NUMERIC,                -- excess_return / sigma. sigma가 NULL이거나 t+h-1이 데이터 범위를 넘으면 NULL

    PRIMARY KEY (ticker, date, window_n, horizon_h),
    CONSTRAINT ck_daily_labels_window_n CHECK (window_n IN (20, 60, 120)),
    CONSTRAINT ck_daily_labels_horizon_h CHECK (horizon_h IN (1, 3, 5, 10))
);

CREATE INDEX IF NOT EXISTS idx_daily_labels_date ON daily_labels (date);
