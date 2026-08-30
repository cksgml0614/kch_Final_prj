-- daily_news_bigkinds 데이터만 비운다(테이블 구조는 유지). 다른 테이블이 이 테이블을 FK로
-- 참조하지 않으므로 다른 테이블에 영향 없음. daily_news와는 완전히 분리된 테이블이라 이 스크립트는
-- daily_news에 영향을 주지 않는다.

TRUNCATE TABLE daily_news_bigkinds RESTART IDENTITY;
