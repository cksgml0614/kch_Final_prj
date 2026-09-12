# 빅카인즈_적재.py
# BigKinds(빅카인즈) CSV/Excel 일괄 적재. daily_news_bigkinds 전용 — 기존 daily_news
# (search_backfill/legacy/finance_crawl)는 전혀 건드리지 않는다(2026-08-29).
# 쓰래기통/Bigkinds.py(레거시, daily_news 대상, 재사용 안 함)를 참고해 새로 작성.
#
# 실행: python -m 빅카인즈.빅카인즈_적재
# DATA_FOLDER 안의 모든 *.csv/*.xlsx/*.xls를 순회한다(정확히 몇 개로 나뉘는지 가정하지 않음).
# .xlsx는 openpyxl, .xls는 xlrd가 필요(둘 다 requirements.txt에 추가, 2026-08-29).

import os
import re
from datetime import datetime

import pandas as pd

from db_manager import get_db_connection

DATA_FOLDER = "./빅카인즈_뉴스"
TARGET_TICKER = "005930"
SOURCE_NAME = "bigkinds"

# BigKinds 내보내기 CSV의 컬럼명 그대로. 이 중 하나라도 없는 파일은 읽기 단계에서 걸러진다.
USECOLS = ["뉴스 식별자", "일자", "언론사", "제목", "통합 분류1", "키워드", "본문", "URL"]

# 큰 파일에서 진행 상황을 볼 수 있도록 일정 행마다 중간 로그를 남긴다.
PROGRESS_EVERY_ROWS = 5000

_DATE_RE = re.compile(r"^\d{8}$")
# 마침표·쉼표·물음표·느낌표·따옴표는 보존 — summary가 본문 전체라 문맥이 중요하다.
# 구 Bigkinds.py(쓰래기통)의 clean_text는 쉼표·마침표만 남기고 다 지웠는데 그보다 완화한 버전.
_ALLOWED_PUNCT_RE = re.compile(r"[^\w\s.,?!'\"“”‘’]")

INSERT_QUERY = """
    INSERT INTO daily_news_bigkinds
        (ticker, date, title, summary, press, article_url, keywords, category, bigkinds_id, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING
    RETURNING id
"""


def clean_text(text):
    """제목/본문 정제. NaN은 빈 문자열로, 그 외엔 허용 문장부호 외 특수문자만 제거."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = _ALLOWED_PUNCT_RE.sub("", text)
    return text.strip()


def _clean_optional(value):
    """
    NaN/빈 문자열은 None으로 변환하고, 그 외엔 앞뒤 공백만 제거해 원문 그대로 보존한다.
    URL·식별자·키워드·카테고리는 clean_text의 특수문자 제거 대상이 아니다(URL이 깨지면 안 됨).
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def _parse_bigkinds_date(raw):
    """일자가 YYYYMMDD 형식인지 확인 후 파싱. 형식이 아니거나 유효하지 않으면 None."""
    if pd.isna(raw):
        return None
    if isinstance(raw, str):
        s = raw.strip()
    else:
        try:
            s = str(int(float(raw)))
        except (TypeError, ValueError):
            return None
    if not _DATE_RE.fullmatch(s):
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def _read_data_file(file_path):
    """
    확장자로 자동 분기해서 읽는다.
    - .csv: cp949 우선 시도 → 실패 시 utf-8-sig 폴백(기존 로직 그대로).
    - .xlsx/.xls: pd.read_excel — 엑셀은 인코딩 문제가 없어 폴백 불필요
      (엔진은 pandas가 확장자 보고 자동 선택: .xlsx→openpyxl, .xls→xlrd).
    반환: (DataFrame, 읽기 방식 라벨) — 라벨은 파일별 처리 로그에 남겨 디버깅에 쓴다.

    ⚠️ 2026-08-29 발견: '뉴스 식별자'는 dtype 지정 없이 읽으면 float64로 자동 추론되는데,
    실제 값이 26자리("02100501.20260829080426001")라 float64 유효자릿수(15~17자리)를
    넘어가 뒷부분이 잘린다. 언론사+날짜 앞자리가 같은 서로 다른 기사가 같은 값으로 뭉개져
    bigkinds_id UNIQUE에서 진짜 다른 기사를 "중복"으로 오판해 버리는 데이터 유실이 발생했다
    (실측: 한 파일에서 66% 행이 충돌, 최대 23개 별개 기사가 한 값으로 뭉개진 사례 확인).
    반드시 dtype=str로 강제해 원문 그대로 보존한다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    id_col = "뉴스 식별자"
    if ext == ".csv":
        try:
            return pd.read_csv(file_path, usecols=USECOLS, dtype={id_col: str}, encoding="cp949"), "csv/cp949"
        except UnicodeDecodeError:
            return pd.read_csv(file_path, usecols=USECOLS, dtype={id_col: str}, encoding="utf-8-sig"), "csv/utf-8-sig"
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path, usecols=USECOLS, dtype={id_col: str}), f"excel({ext})"
    else:
        raise ValueError(f"지원하지 않는 확장자: {ext}")


def _print_verification(conn, stats):
    print("\n=== 적재 요약 ===", flush=True)
    print(f"  전체 읽은 행 수: {stats['read']}", flush=True)
    print(f"  실제 적재된 행 수(신규): {stats['inserted']}", flush=True)
    print(f"  중복(ON CONFLICT)으로 건너뛴 행 수: {stats['duplicate']}", flush=True)
    print(f"  파싱/처리 실패로 스킵된 행 수: {stats['skipped']}", flush=True)
    print(f"  title에 '삼성전자' 미포함 행 수: {stats['no_samsung']}", flush=True)
    if stats["no_samsung_samples"]:
        print(f"    예시(최대 5건): {stats['no_samsung_samples']}", flush=True)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_news_bigkinds")
        total_rows, min_date, max_date = cur.fetchone()
        print("\n=== daily_news_bigkinds 테이블 현황 ===", flush=True)
        print(f"  총 행 수: {total_rows}", flush=True)
        print(f"  날짜 범위: {min_date} ~ {max_date}", flush=True)

        cur.execute(
            """
            SELECT press, COUNT(*) AS cnt
            FROM daily_news_bigkinds
            GROUP BY press
            ORDER BY cnt DESC
            LIMIT 10
            """
        )
        rows = cur.fetchall()
        print("  언론사별 건수 상위 10:", flush=True)
        for press, cnt in rows:
            print(f"    {press}: {cnt}건", flush=True)


def load_csv_to_db(folder_path, ticker):
    if not os.path.exists(folder_path):
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}", flush=True)
        return

    files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in (".csv", ".xlsx", ".xls")
    )
    if not files:
        print("🔍 폴더 안에 CSV/Excel 파일이 없습니다.", flush=True)
        return

    print(f"📂 총 {len(files)}개의 파일(CSV/Excel)을 발견했습니다. 적재를 시작합니다.", flush=True)

    conn = get_db_connection()
    if conn is None:
        return

    stats = {
        "read": 0, "inserted": 0, "duplicate": 0, "skipped": 0,
        "no_samsung": 0, "no_samsung_samples": [],
    }

    try:
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            ext = os.path.splitext(file_name)[1].lower()
            print(f"🚀 {file_name} 처리 중... (확장자 {ext})", flush=True)

            try:
                df, read_method = _read_data_file(file_path)
            except Exception as e:
                print(f"   ❌ 파일을 읽을 수 없습니다: {file_name} — {e}", flush=True)
                continue

            file_read = len(df)
            file_inserted = 0
            file_duplicate = 0
            file_skipped = 0

            with conn.cursor() as cur:
                for i, (_, row) in enumerate(df.iterrows(), 1):
                    parsed_date = _parse_bigkinds_date(row["일자"])
                    if parsed_date is None:
                        file_skipped += 1
                        continue

                    title = clean_text(row["제목"])
                    summary = clean_text(row["본문"])
                    press = _clean_optional(row["언론사"])
                    category = _clean_optional(row["통합 분류1"])
                    keywords = _clean_optional(row["키워드"])
                    bigkinds_id = _clean_optional(row["뉴스 식별자"])
                    article_url = _clean_optional(row["URL"])

                    if "삼성전자" not in title:
                        stats["no_samsung"] += 1
                        if len(stats["no_samsung_samples"]) < 5:
                            stats["no_samsung_samples"].append(title)

                    try:
                        with conn.transaction():
                            cur.execute(
                                INSERT_QUERY,
                                (
                                    ticker, parsed_date, title, summary, press, article_url,
                                    keywords, category, bigkinds_id, SOURCE_NAME,
                                ),
                            )
                            result = cur.fetchone()
                        if result is not None:
                            file_inserted += 1
                        else:
                            file_duplicate += 1
                    except Exception as e:
                        file_skipped += 1
                        print(f"      ❌ 행 삽입 실패(bigkinds_id={bigkinds_id!r}): {e}", flush=True)

                    if i % PROGRESS_EVERY_ROWS == 0:
                        print(f"      ... {i}/{file_read}행 처리 중", flush=True)

            conn.commit()
            stats["read"] += file_read
            stats["inserted"] += file_inserted
            stats["duplicate"] += file_duplicate
            stats["skipped"] += file_skipped
            print(
                f"   ✅ {file_name} [{read_method}]: 읽음 {file_read} / 신규 {file_inserted} / "
                f"중복 {file_duplicate} / 스킵 {file_skipped}",
                flush=True,
            )
    finally:
        _print_verification(conn, stats)
        conn.close()


if __name__ == "__main__":
    batch_start_time = datetime.now()
    load_csv_to_db(DATA_FOLDER, TARGET_TICKER)
    print(f"⏱️ 총 소요 시간: {datetime.now() - batch_start_time}", flush=True)
