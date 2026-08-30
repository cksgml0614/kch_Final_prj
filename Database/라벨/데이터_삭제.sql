-- daily_labels 데이터만 비운다(테이블 구조는 유지). 다른 테이블이 daily_labels를 FK로 참조하지
-- 않으므로 다른 테이블에 영향 없음.

TRUNCATE TABLE daily_labels;
