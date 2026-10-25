-- model_predictions 위의 뷰 2개를 삭제한다. 뷰만 지우고 원본 테이블·데이터는 건드리지 않는다. 두 뷰에 의존하는 객체는 없다.

DROP VIEW IF EXISTS model_predictions_trusted;

DROP VIEW IF EXISTS model_predictions_realized;
