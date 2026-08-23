# config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # DB 설정
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")

    DB_URL = f"host={DB_HOST} dbname={DB_NAME} user={DB_USER} password={DB_PASS} port={DB_PORT}"

    # API 설정
    NAVER_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    ECOS_API_KEY = os.getenv("ECOS_API_KEY")