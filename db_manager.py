# db_manager.py
import psycopg
from config import Config

def get_db_connection():
    """ DB 연결 객체를 반환합니다."""
    # NEON_DB_URL이 없을 때 psycopg.connect(None)을 그대로 부르면 libpq 기본값(localhost)으로
    # 조용히 접속을 시도한다. 그래서 먼저 막는다.
    if not Config.DB_URL:
        print("❌ DB 연결 실패: NEON_DB_URL이 설정되지 않았습니다(.env 또는 Actions secrets 확인)")
        return None
    try:
        conn = psycopg.connect(Config.DB_URL)
        return conn
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None