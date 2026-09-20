# config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # DB 설정 — 로컬(도커 Postgres) 기본값. 클라우드 이전 후에도 fallback으로 계속 유지한다.
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")

    _LOCAL_DB_URL = f"host={DB_HOST} dbname={DB_NAME} user={DB_USER} password={DB_PASS} port={DB_PORT}"

    # USE_CLOUD_DB=true면 클라우드(Neon, pooled 연결)를 쓴다(2026-09-20, DB 클라우드 이전).
    # NEON_DB_URL(pooled)만 쓴다 — db_manager.get_db_connection()이 요청마다 커넥션을 새로
    # 열고 닫는 패턴이라 pooler가 필요하다. NEON_DB_URL_DIRECT(비-pooled)는 세션 단위 SET
    # 문이 필요한 pg_dump/pg_restore 같은 마이그레이션 작업 전용이며 애플리케이션 코드에서는
    # 쓰지 않는다. 기본값 false — 이 값을 안 주면 기존과 동일하게 로컬 DB를 그대로 쓴다.
    USE_CLOUD_DB = os.getenv("USE_CLOUD_DB", "false").strip().lower() == "true"
    DB_URL = os.getenv("NEON_DB_URL") if USE_CLOUD_DB else _LOCAL_DB_URL

    # API 설정
    NAVER_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    ECOS_API_KEY = os.getenv("ECOS_API_KEY")