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


def crawl_and_save_naver_news(stock_name, ticker, volatile_dates):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://www.naver.com/',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    conn = get_db_connection()
    if not conn: return

    try:
        volatile_dates = sorted(volatile_dates, reverse=True)

        for target_date in volatile_dates:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')

            # 💡 [핵심] 수집할 날짜 리스트를 개별적으로 생성 (D-1, D)
            days_to_search = [
                (target_dt - timedelta(days=1)).strftime('%Y.%m.%d'),  # 어제
                target_dt.strftime('%Y.%m.%d')  # 오늘
            ]

            print(f"🚀 [{target_date}] 변동성 분석을 위해 이틀치 데이터를 개별 수집합니다.")

            for search_date in days_to_search:
                print(f"   🔎 {search_date} 일자 뉴스 수집 중...")

                # 시작일(ds)과 종료일(de)을 동일하게 설정하여 '해당 날짜'만 정밀 타격
                url = f"https://search.naver.com/search.naver?where=news&query={stock_name}&pd=3&ds={search_date}&de={search_date}"

                res = requests.get(url, headers=headers, timeout=15)
                if len(res.text) < 10000:
                    print(f"      ❌ 차단 혹은 로딩 실패 (길이: {len(res.text)})")
                    continue

                soup = BeautifulSoup(res.text, 'html.parser')
                title_links = soup.find_all('a', attrs={'data-heatmap-target': '.tit'})
                if not title_links: title_links = soup.select('a.news_tit')

                news_count = 0
                with conn.cursor() as cur:
                    for link in title_links:
                        if news_count >= 25: break  # 하루당 25건, 이틀 합쳐 약 50건 수집

                        title = link.get_text(strip=True)
                        parent_area = link.find_parent('div')
                        summary_elem = None
                        if parent_area:
                            summary_elem = parent_area.find_next_sibling('div')
                            if not summary_elem:
                                summary_elem = parent_area.select_one('[class*="body"]')
                        summary = summary_elem.get_text(strip=True) if summary_elem else title

                        # 💡 [라벨링 매칭] search_date가 언제든,
                        # DB에는 변동성 발생일인 'target_date'로 저장하여 주가와 매칭시킵니다.
                        cur.execute("""
                                    INSERT INTO daily_news (ticker, date, title, summary)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT DO NOTHING;
                                    """, (ticker, target_date, title, summary))
                        news_count += 1

                conn.commit()
                print(f"      ✅ {search_date} 데이터 {news_count}건 저장 완료")

                # 날짜 간 요청 사이에 짧은 휴식
                time.sleep(random.uniform(0.7, 1.5))

            # 한 세트(D-1, D) 완료 후 휴식
            time.sleep(random.uniform(1.0, 2.0))

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    stock_code = "005930"
    volatile_dates = get_significant_volatility_dates(stock_code, 0.03)
    if volatile_dates:
        crawl_and_save_naver_news("삼성전자", stock_code, volatile_dates)