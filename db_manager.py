# db_manager.py — 모든 모듈이 쓰는 DB 연결 함수 하나(get_db_connection). 접속 정보는 config.Config.DB_URL.
# 배경: 결정기록.md의 "DB 연결 실패는 None이 아니라 예외로 알린다"
import psycopg
from config import Config

def get_db_connection():
    """
    설명: Neon DB에 새 psycopg 연결을 열어 돌려준다.
    주의 사항: 접속 문자열이 없거나 접속에 실패하면 원인을 담은 RuntimeError를 던진다(None을 돌려주지 않는다 —
    호출부의 `if not conn` 분기는 도달하지 않는다). 연결을 닫는 것은 호출자 몫(with 블록 등).
    입력 타입->반환 타입: () -> psycopg.Connection
    """
    # NEON_DB_URL이 없을 때 psycopg.connect(None)을 그대로 부르면 libpq 기본값(localhost)으로
    # 조용히 접속을 시도한다. 그래서 먼저 막는다.
    if not Config.DB_URL:
        raise RuntimeError("DB 연결 실패: NEON_DB_URL이 설정되지 않았습니다(.env 또는 Actions secrets 확인)")
    try:
        return psycopg.connect(Config.DB_URL)
    except psycopg.Error as e:
        raise RuntimeError(f"DB 연결 실패: {e}") from e
