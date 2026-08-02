import FinanceDataReader as fdr
from datetime import timedelta, date
from db_manager import get_db_connection

TICKERS = [
    '005930',  # 삼성전자
    '000660',  # SK하이닉스
]


def get_last_date(cur, ticker):
    """DB에서 해당 종목의 가장 최신 날짜를 가져옵니다."""
    query = "SELECT MAX(date) FROM daily_stock_prices WHERE ticker = %s"
    cur.execute(query, (ticker,))
    result = cur.fetchone()[0]
    return result


def update_stock_data(ticker):
    with get_db_connection() as conn:  # 중앙 매니저에게 연결을 요청
        if not conn:
            return "실패 (DB 연결 없음)"
        with conn.cursor() as cur:
            # 1. 마지막으로 저장된 날짜 확인
            last_date = get_last_date(cur, ticker)

            if last_date:
                # 데이터가 있다면 마지막 날짜 '다음 날'부터 수집
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"🔄 {ticker}: {last_date}까지 데이터가 있네요. {start_date}부터 업데이트를 시작합니다.")
            else:
                # 데이터가 아예 없다면 2020년부터 전체 수집
                start_date = '2020-01-01'
                print(f"🆕 {ticker}: 저장된 데이터가 없습니다. {start_date}부터 전체 적재를 시작합니다.")

            # 오늘 날짜와 비교해서 이미 최신이면 종료
            if last_date == date.today():
                print(f"✅ {ticker}: 이미 최신 데이터입니다.")
                return "스킵 (이미 최신)"

            # 2. 필요한 만큼만 데이터 수집
            df = fdr.DataReader(ticker, start_date)

            if df.empty:
                print(f"📍 {ticker}: 새로 추가할 데이터가 없습니다 (주말/휴장일 등).")
                return "스킵 (신규 데이터 없음)"


            # 3. 데이터 가공 및 적재 (기존 로직 동일)
            df = df.reset_index()
            df['ticker'] = ticker
            df = df.fillna(0)

            data_list = [
                (row['ticker'], row['Date'].date(), int(row['Open']), int(row['High']),
                 int(row['Low']), int(row['Close']), int(row['Volume']), float(row['Change']))
                for _, row in df.iterrows()
            ]

            insert_query = """
                           INSERT INTO daily_stock_prices (ticker, date, open, high, low, close, volume, change_rate)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (ticker, date) DO NOTHING; \
                           """
            cur.executemany(insert_query, data_list)
            print(f"🚀 {ticker}: {len(data_list)}건의 새로운 데이터 적재 완료!")
            return f"완료 ({len(data_list)}건)"


if __name__ == "__main__":
    results = {}
    for ticker in TICKERS:
        try:
            results[ticker] = update_stock_data(ticker)
        except Exception as e:
            print(f"❌ {ticker}: 업데이트 실패: {e}")
            results[ticker] = f"실패 ({e})"

    print("\n=== 처리 결과 요약 ===")
    for ticker, status in results.items():
        print(f"{ticker}: {status}")