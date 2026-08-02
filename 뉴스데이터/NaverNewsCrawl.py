import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
from db_manager import get_db_connection

SEARCH_YEAR = 3


def get_significant_volatility_dates(ticker="005930", threshold=0.03):
    """최근 SEARCH_YEAR년 동안 등락률이 ±threshold 이상인 날짜 리스트 반환"""
    conn = get_db_connection()
    if not conn: return []
    try:
        n_years_ago = (datetime.now() - timedelta(days=SEARCH_YEAR * 365)).date()
        query = """
                SELECT date, change_rate \
                FROM daily_stock_prices
                WHERE ticker = %s \
                  AND date >= %s \
                  AND (change_rate >= %s OR change_rate <= -%s)
                ORDER BY date DESC \
                """
        with conn.cursor() as cur:
            cur.execute(query, (ticker, n_years_ago, threshold, threshold))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        if df.empty: return []
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].dt.strftime('%Y-%m-%d').tolist()
    except Exception as e:
        print(f"❌ 데이터 분석 중 오류 발생: {e}");
        return []
    finally:
        conn.close()


def build_search_plan(volatile_dates):
    """
    변동성 날짜 리스트로부터 '실제 크롤링할 날짜(search_date) -> 라벨로 쓸 날짜(target_date)'
    매핑을 만든다.

    변동성 날짜가 연속되면(예: 3/10, 3/11 모두 변동성 큼) 3/10이 '3/11의 전날'로도,
    '3/10 자기 자신'으로도 필요해져서 같은 날짜를 두 번 크롤링하고 두 개의 date로
    중복 저장되는 문제가 생긴다. search_date 하나당 target_date를 하나만 배정해서 이를 막는다.

    우선순위: search_date 자신이 변동성 날짜면 자기 자신에 매핑(동일자 매칭 최우선),
    그렇지 않고 다른 변동성 날짜의 '전날'로만 필요한 경우에는 그 다음날에 매핑한다.
    """
    volatile_set = set(volatile_dates)
    plan = {}
    for d in volatile_dates:
        d_dt = datetime.strptime(d, '%Y-%m-%d')
        prev_day = (d_dt - timedelta(days=1)).strftime('%Y-%m-%d')

        plan[d] = d  # 동일자 매칭은 항상 최우선(무조건 덮어씀)
        if prev_day not in volatile_set:
            plan.setdefault(prev_day, d)  # 이미 자기 자신으로 배정된 날짜는 건드리지 않음

    return plan


def crawl_and_save_naver_news(stock_name, ticker, volatile_dates):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://www.naver.com/',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    conn = get_db_connection()
    if not conn: return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT date FROM daily_news WHERE ticker = %s", (ticker,))
            loaded_dates = {row[0].strftime('%Y-%m-%d') for row in cur.fetchall()}

        target_dates = [d for d in volatile_dates if d not in loaded_dates]
        skipped = len(volatile_dates) - len(target_dates)
        if skipped:
            print(f"⏭️  이미 적재된 날짜 {skipped}개 건너뜀")

        if not target_dates:
            print("✨ 모든 변동성 날짜가 이미 적재되어 있습니다.")
            return

        plan = build_search_plan(target_dates)
        search_dates = sorted(plan.keys(), reverse=True)

        print(f"🚀 변동성 날짜 {len(target_dates)}개(미적재) -> 중복 제거 후 실제 크롤링 {len(search_dates)}일")

        for search_date in search_dates:
            target_date = plan[search_date]
            search_date_fmt = datetime.strptime(search_date, '%Y-%m-%d').strftime('%Y.%m.%d')

            print(f"   🔎 {search_date} 일자 뉴스 수집 중... (라벨: {target_date})")

            # 시작일(ds)과 종료일(de)을 동일하게 설정하여 '해당 날짜'만 정밀 타격
            url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=3&ds={search_date_fmt}&de={search_date_fmt}"

            res = requests.get(url, headers=headers, timeout=15)
            if len(res.text) < 10000:
                print(f"      ❌ 차단 혹은 로딩 실패 (길이: {len(res.text)})")
                time.sleep(random.uniform(0.7, 1.5))
                continue

            soup = BeautifulSoup(res.text, 'html.parser')
            title_links = soup.find_all('a', attrs={'data-heatmap-target': '.tit'})
            if not title_links: title_links = soup.select('a.news_tit')

            news_count = 0
            with conn.cursor() as cur:
                for link in title_links:
                    if news_count >= 25: break  # 검색일당 최대 25건 수집

                    title = link.get_text(strip=True)
                    parent_area = link.find_parent('div')
                    summary_elem = None
                    if parent_area:
                        summary_elem = parent_area.find_next_sibling('div')
                        if not summary_elem:
                            summary_elem = parent_area.select_one('[class*="body"]')
                    summary = summary_elem.get_text(strip=True) if summary_elem else title

                    # 💡 [라벨링 매칭] search_date가 언제든,
                    # DB에는 계획에서 배정된 target_date로 저장하여 주가와 매칭시킵니다.
                    cur.execute("""
                                INSERT INTO daily_news (ticker, date, title, summary)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (ticker, date, title, summary) DO NOTHING;
                                """, (ticker, target_date, title, summary))
                    news_count += 1

            conn.commit()
            print(f"      ✅ {search_date} 데이터 {news_count}건 저장 완료 (date={target_date})")

            # 요청 간 짧은 휴식
            time.sleep(random.uniform(0.7, 1.5))

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    stock_code = "005930"
    volatile_dates = get_significant_volatility_dates(stock_code, 0.02)
    if volatile_dates:
        crawl_and_save_naver_news("삼성전자", stock_code, volatile_dates)