# db_manager.py
import psycopg
from config import Config

def get_db_connection():
    """ DB 연결 객체를 반환합니다."""
    try:
        conn = psycopg.connect(Config.DB_URL)
        return conn
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None