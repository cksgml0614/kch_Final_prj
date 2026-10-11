# config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # DB는 클라우드(Neon) 하나만 쓴다. 2026-10-03에 로컬 Docker Postgres fallback과
    # USE_CLOUD_DB 토글을 없앴다(로컬 DB 폐기, 덤프는 저장소 밖에 보관).
    # NEON_DB_URL(pooled)만 쓴다 — db_manager.get_db_connection()이 요청마다 커넥션을 새로
    # 열고 닫는 패턴이라 pooler가 필요하다. NEON_DB_URL_DIRECT(비-pooled)는 세션 단위 SET
    # 문이 필요한 pg_dump/pg_restore 같은 마이그레이션 작업 전용이며 애플리케이션 코드에서는
    # 쓰지 않는다.
    DB_URL = os.getenv("NEON_DB_URL")

    # API 설정
    ECOS_API_KEY = os.getenv("ECOS_API_KEY")


def redact(text):
    """로그·상태 문자열에서 비밀값을 ***로 바꾼다(2026-10-11). 예외 메시지를 출력·저장하기 전에 쓴다."""
    s = str(text)
    for secret in (Config.ECOS_API_KEY, Config.DB_URL):
        if secret:
            s = s.replace(secret, "***")
    return s