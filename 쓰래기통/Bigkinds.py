import pandas as pd
import os
import re
import psycopg
from datetime import datetime
from db_manager import get_db_connection

# 1. 설정
DATA_FOLDER = "./Data/News"  # CSV 파일들을 넣어둘 폴더 경로
TARGET_TICKER = "005930"  # 적재할 종목 코드 (삼성전자)


def clean_text(text):
    """텍스트 내 불필요한 특수문자 제거"""
    if pd.isna(text): return ""
    text = str(text)
    # 특수문자 제거 (KoBERT 학습에 방해되는 기호들 정리)
    text = re.sub(r'[^\w\s\d,.]', '', text)
    return text.strip()


def load_csv_to_db(folder_path, ticker):
    # 폴더 내 CSV 파일 목록 확보
    if not os.path.exists(folder_path):
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not files:
        print("🔍 폴더 안에 CSV 파일이 없습니다.")
        return

    print(f"📂 총 {len(files)}개의 CSV 파일을 발견했습니다. 적재를 시작합니다.")

    # DB 인서트 쿼리 (우리 테이블 구조에 맞춤)
    insert_query = """
                   INSERT INTO daily_news (ticker, date, title, summary)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT DO NOTHING; \
                   """

    total_inserted = 0

    with get_db_connection() as conn:
        if not conn: return

        with conn.cursor() as cur:
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                print(f"🚀 {file_name} 처리 중...")

                try:
                    # 💡 빅카인즈 CSV는 보통 utf-8-sig 혹은 cp949 인코딩입니다.
                    # 필요한 컬럼만 읽어서 메모리 사용량 최적화
                    try:
                        # 1순위: 한국 윈도우 표준 인코딩 시도
                        df = pd.read_csv(file_path, usecols=['일자', '제목', '키워드'], encoding='cp949')
                    except UnicodeDecodeError:
                        try:
                            # 2순위: UTF-8 (BOM 포함) 시도
                            df = pd.read_csv(file_path, usecols=['일자', '제목', '키워드'], encoding='utf-8-sig')
                        except Exception as e:
                            print(f"   ❌ 인코딩 오류로 파일을 읽을 수 없습니다: {file_name}")
                            continue

                    data_to_insert = []
                    for _, row in df.iterrows():
                        try:
                            # 날짜 변환 (20240425 -> 2024-04-25)
                            raw_date = str(row['일자'])
                            formatted_date = datetime.strptime(raw_date, '%Y%m%d').date()

                            title = clean_text(row['제목'])
                            summary = clean_text(row['키워드'])

                            data_to_insert.append((ticker, formatted_date, title, summary))
                        except Exception as e:
                            continue  # 날짜 파싱 에러 등 개별 행 에러 시 스킵

                    # Bulk Insert 실행
                    if data_to_insert:
                        cur.executemany(insert_query, data_to_insert)
                        conn.commit()  # 파일 단위로 커밋하여 안정성 확보
                        total_inserted += len(data_to_insert)
                        print(f"   ✅ {len(data_to_insert)}건 적재 성공 (현재 누적: {total_inserted}건)")

                except Exception as e:
                    print(f"   ❌ 파일 처리 중 에러 발생 ({file_name}): {e}")

    print(f"\n✨ 모든 작업 완료! 총 {total_inserted}건의 데이터가 DB에 저장되었습니다.")


if __name__ == "__main__":
    # 실행 전 bigkinds_data 폴더에 CSV 파일을 넣었는지 확인하세요!
    batch_start_time = datetime.now()
    load_csv_to_db(DATA_FOLDER, TARGET_TICKER)
    print(f"⏱️ 총 소요 시간: {datetime.now() - batch_start_time}")