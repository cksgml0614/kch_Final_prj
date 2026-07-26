import requests
import re
import psycopg
from datetime import datetime
from db_manager import get_db_connection
from config import Config

# # 1. 설정 및 인증

CLIENT_ID = Config.NAVER_ID
CLIENT_SECRET = Config.NAVER_SECRET



# 종목 및 관련 키워드 설정 (확장성 고려)
STOCK_TARGETS = {
    '005930': {'name': '삼성전자', 'keywords': ['이재용', '반도체', 'HBM', '실적', '갤럭시']},
    '000660': {'name': 'SK하이닉스', 'keywords': ['HBM', '엔비디아', '반도체', '메모리']}
}
NEWS_COUNT=100

def clean_text(text):
    """[기능 1] API 텍스트 내 HTML 태그 및 엔티티 제거"""
    if not text: return ""
    text = re.sub('<.*?>', '', text)  # <b> 등 태그 제거
    text = re.sub('&quot;', '"', text)
    text = re.sub('&amp;', '&', text)
    text = re.sub('&lt;', '<', text)
    text = re.sub('&gt;', '>', text)
    return text.strip()


def is_relevant_news(title, summary, stock_info):
    """[기능 2] 제목과 요약본 기반 관련성 검증 (순도 체크)"""
    name = stock_info['name']
    keywords = stock_info['keywords']

    # 1. 제목에 종목명이 있으면 최우선 통과
    if name in title:
        return True

    # 2. 요약본 내 종목명 및 연관 키워드 빈도 점수 계산
    score = summary.count(name) * 2
    for kw in keywords:
        score += summary.count(kw)

    # 요약본 기준이므로 점수 기준을 3점 정도로 낮추어 유연하게 적용
    return score >= 3


def collect_and_store_news(ticker):
    stock_info = STOCK_TARGETS[ticker]
    query = stock_info['name']

    # 💡 오늘 날짜 가져오기 (실행 시점 기준)
    today = datetime.now().date()

    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={NEWS_COUNT}&sort=date"  # 💡 정렬을 'date'로 변경 권장
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return

        items = response.json().get('items', [])
        refined_data = []

        for item in items:
            # 1. 날짜 먼저 확인하여 '오늘' 기사가 아니면 바로 다음으로 넘어감 (성능 최적화)
            pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z').date()

            if pub_date != today:
                continue  # 오늘 날짜가 아니면 skip

            # 2. 오늘 날짜인 경우에만 텍스트 정제 및 관련성 검사 진행
            title = clean_text(item['title'])
            summary = clean_text(item['description'])

            if is_relevant_news(title, summary, stock_info):
                refined_data.append((ticker, pub_date, title, summary))

        # [기능 3] DB 적재
        if not refined_data:
            print(f"ℹ️ {stock_info['name']}: 오늘 새로 올라온 관련 뉴스가 없습니다.")
            return

        insert_query = """
                       INSERT INTO daily_news (ticker, date, title, summary)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT DO NOTHING; \
                       """

        with get_db_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.executemany(insert_query, refined_data)
                    conn.commit()
                print(f"✅ {stock_info['name']}: 오늘자 뉴스 {len(refined_data)}건 적재 완료!")


    except Exception as e:
        print(f"❌ 에러 발생: {e}")


if __name__ == "__main__":
    for ticker in STOCK_TARGETS.keys():
        collect_and_store_news(ticker)