# db_manager.py
import psycopg
from config import Config

def get_db_connection():
    """DB 연결 객체를 반환한다. 실패하면 원인을 담은 RuntimeError를 던진다.

    2026-10-05: 예전에는 실패 시 None을 반환했지만, 모든 호출부가 `with get_db_connection() as conn:`
    형태라 None이면 `if not conn` 검사에 닿기 전에 `with None`이 TypeError("'NoneType' object does
    not support the context manager protocol")를 내 원인(예: Neon quota 초과)이 가려졌다. 호출부의
    `if not conn` 분기는 이제 도달하지 않는 코드다(남겨둬도 무해)."""
    # NEON_DB_URL이 없을 때 psycopg.connect(None)을 그대로 부르면 libpq 기본값(localhost)으로
    # 조용히 접속을 시도한다. 그래서 먼저 막는다.
    if not Config.DB_URL:
        raise RuntimeError("DB 연결 실패: NEON_DB_URL이 설정되지 않았습니다(.env 또는 Actions secrets 확인)")
    try:
        return psycopg.connect(Config.DB_URL)
    except psycopg.Error as e:
        raise RuntimeError(f"DB 연결 실패: {e}") from e
