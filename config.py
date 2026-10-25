# config.py — 비밀값(DB 접속 문자열·ECOS API 키)을 환경변수(.env, Actions secrets)에서 읽는 설정과 출력용 비밀값 가리기(redact).
# 코드에 비밀값을 직접 쓰지 않는다. 배경: 결정기록.md의 "DB는 Neon pooled 접속 하나만 쓴다", "비밀값은 출력 전에 가린다"
import os
from urllib.parse import quote, quote_plus

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # DB는 Neon 하나, pooled 접속(NEON_DB_URL)만 쓴다. NEON_DB_URL_DIRECT(비-pooled)는 pg_dump 등 수동 작업 전용.
    DB_URL = os.getenv("NEON_DB_URL")

    # API 설정
    ECOS_API_KEY = os.getenv("ECOS_API_KEY")


def redact(text):
    """
    설명: 문자열(또는 예외)에서 ECOS 키·DB 접속 문자열과 그 URL 인코딩 변형을 ***로 바꿔 돌려준다.
    주의 사항: 예외 메시지를 출력·저장하기 전에 쓴다. 긴 변형부터 치환한다. 비밀값이 비어 있으면 그 항목은 건너뛴다.
    입력 타입->반환 타입: (object) -> str
    """
    s = str(text)
    for secret in (Config.ECOS_API_KEY, Config.DB_URL):
        if not secret:
            continue
        # URL 인코딩 변형까지 가린다(긴 것부터 치환). requests는 대문자 %XX를 쓴다.
        for v in sorted({secret, quote(secret), quote(secret, safe=""), quote_plus(secret)}, key=len, reverse=True):
            s = s.replace(v, "***")
    return s