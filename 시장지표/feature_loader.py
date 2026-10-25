# feature_loader.py — 종목 OHLCV에 시장지표 9종을 미래 정보 누수 없이 붙인 피처 행렬과, 가격·지표 일괄 조회 캐시.
# 파이프라인 단계: 피처 입력(DB 읽기만). 주요 진입 함수: get_features, load_price_cache, load_indicator_cache.
# 각 지표는 거래일 D에 published_date < D인 값 중 최신값만 붙는다(같은 날 공표분 제외).
# 배경: 결정기록.md의 "지표는 published_date < 거래일로만 붙인다", "가격·지표는 실행당 한 번 읽어 캐시로 공유"

import sys

import pandas as pd

from db_manager import get_db_connection

# Windows 콘솔 기본 코드페이지(cp949)는 이모지/특수문자를 인코딩하지 못해 죽는다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 피처로 붙이는 지표 9종(지수·환율 3종 + 거시지표 6종). 순서가 피처 컬럼 순서다.
INDICATOR_CODES = [
    "KOSPI",
    "KOSDAQ",
    "USD_KRW",
    "ECOS_722Y001_0101000",
    "ECOS_817Y002_010200000",
    "ECOS_817Y002_010210000",
    "ECOS_901Y009_0",
    "ECOS_161Y005_BBHS00",
    "ECOS_901Y067_I16E",
]


def _load_prices(cur, ticker, start_date, end_date):
    """
    설명: daily_stock_prices에서 종목의 [start_date, end_date] OHLCV를 날짜순 DataFrame으로 읽는다.
    주의 사항: 호출자의 커서를 쓴다. 변환은 _prices_frame과 같다.
    입력 타입->반환 타입: (psycopg.Cursor, str, str, str) -> pd.DataFrame
    """
    cur.execute(
        """
        SELECT date, open, high, low, close, volume
        FROM daily_stock_prices
        WHERE ticker = %s AND date BETWEEN %s AND %s
        ORDER BY date
        """,
        (ticker, start_date, end_date),
    )
    return _prices_frame(cur.fetchall())


def _prices_frame(rows):
    """
    설명: (date, open, high, low, close, volume) 행 목록을 float 값·datetime date 컬럼의 DataFrame으로 바꾼다.
    주의 사항: _load_prices와 load_price_cache가 같은 변환을 쓰려고 분리했다. 빈 입력은 빈 DataFrame.
    입력 타입->반환 타입: (list[tuple]) -> pd.DataFrame
    """
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    # NUMERIC은 psycopg가 Decimal로 돌려줘 pandas 연산에서 float과 충돌하므로 먼저 float으로 바꾼다
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_price_cache(tickers, start_date, end_date):
    """
    설명: 여러 종목의 OHLCV를 쿼리 한 번(ticker = ANY)으로 읽어 {ticker: DataFrame}(_load_prices와 같은 스키마)으로 돌려준다.
    주의 사항: 각 DataFrame의 attrs["span"]에 (start_date, end_date)를 남겨 get_features가 다른 구간에 쓰면 예외를 낸다.
    행이 없는 종목도 빈 DataFrame으로 들어간다.
    입력 타입->반환 타입: (list[str], str, str) -> dict[str, pd.DataFrame]
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, date, open, high, low, close, volume
                FROM daily_stock_prices
                WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
                ORDER BY ticker, date
                """,
                (list(tickers), start_date, end_date),
            )
            rows = cur.fetchall()
    rows_by_ticker = {t: [] for t in tickers}
    for r in rows:
        rows_by_ticker[r[0]].append(r[1:])
    span = (str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date()))
    cache = {}
    for t, t_rows in rows_by_ticker.items():
        df = _prices_frame(t_rows)
        df.attrs["span"] = span
        cache[t] = df
    return cache


def _load_indicator_series(cur, indicator_code, end_date, start_date=None):
    """
    설명: 지표 하나의 published_date <= end_date 행(ref_date, value, pub_date)을 pub_date 오름차순 DataFrame으로 읽는다.
    주의 사항: start_date를 주면 "published_date < start_date인 마지막 공표일" 이상만 읽는다 — _asof_join 결과는 하한 없을 때와 같다
    (그 앞 행은 어느 거래일에도 붙지 않음). 시작일 이전 행이 없으면 전부 읽는다. value는 바로 float으로 바꾼다.
    입력 타입->반환 타입: (psycopg.Cursor, str, str, str|None) -> pd.DataFrame
    """
    if start_date is None:
        cur.execute(
            """
            SELECT date AS ref_date, value, published_date AS pub_date
            FROM market_indicators
            WHERE indicator_code = %s AND published_date <= %s
            ORDER BY published_date
            """,
            (indicator_code, end_date),
        )
    else:
        cur.execute(
            """
            SELECT date AS ref_date, value, published_date AS pub_date
            FROM market_indicators
            WHERE indicator_code = %s AND published_date <= %s
              AND published_date >= COALESCE(
                  (SELECT MAX(published_date) FROM market_indicators
                   WHERE indicator_code = %s AND published_date < %s),
                  '-infinity'::date)
            ORDER BY published_date
            """,
            (indicator_code, end_date, indicator_code, start_date),
        )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["ref_date", "value", "pub_date"])
    if df.empty:
        return df
    df["value"] = df["value"].astype(float)  # NUMERIC(Decimal)은 가장 먼저 float으로
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["ref_date"] = pd.to_datetime(df["ref_date"])
    return df.sort_values("pub_date")


def _asof_join(trading_dates, indicator_df):
    """
    설명: 거래일 D마다 published_date < D인 지표 값 중 최신값을 붙인 Series(인덱스 = trading_dates)를 돌려준다.
    주의 사항: 같은 날 공표분은 붙지 않는다(allow_exact_matches=False). 월별 지표는 다음 공표까지 같은 값이 이어지고,
    첫 공표 전과 빈 지표는 NaN.
    입력 타입->반환 타입: (pd.Series, pd.DataFrame) -> pd.Series
    """
    if indicator_df.empty:
        return pd.Series([float("nan")] * len(trading_dates), index=trading_dates)

    left = pd.DataFrame({"date": trading_dates}).sort_values("date")
    right = indicator_df[["pub_date", "value"]].sort_values("pub_date")
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="pub_date",
        direction="backward", allow_exact_matches=False,
    )
    return merged.set_index("date")["value"].reindex(trading_dates)


def load_indicator_cache(end_date, start_date=None):
    """
    설명: INDICATOR_CODES 9종을 커넥션 하나로 읽어 {code: DataFrame(_load_indicator_series 스키마)}로 돌려준다.
    주의 사항: ⚠️ end_date 일치는 검사하지 않는다 — get_features에 넘길 때 같은 end_date로 만든 캐시인지는 호출자가 책임진다.
    start_date를 주면 attrs["start_date"]에 남겨, 그보다 이른 start_date로 쓰면 get_features가 예외를 낸다.
    입력 타입->반환 타입: (str, str|None) -> dict[str, pd.DataFrame]
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            cache = {code: _load_indicator_series(cur, code, end_date, start_date) for code in INDICATOR_CODES}
    if start_date is not None:
        for df in cache.values():
            df.attrs["start_date"] = str(pd.Timestamp(start_date).date())
    return cache


def get_features(ticker, start_date, end_date, precomputed_indicators=None, precomputed_prices=None):
    """
    설명: 종목 OHLCV(인덱스 = 거래일)에 INDICATOR_CODES 9개 컬럼(published_date < 거래일인 최신값)을 붙인 피처 행렬을 돌려준다.
    주의 사항: precomputed_indicators(load_indicator_cache)·precomputed_prices(load_price_cache[ticker])를 주면 그 쪽 DB 조회를 건너뛴다.
    결과는 캐시 없이 읽을 때와 같고, 둘 다 주면 DB에 접속하지 않는다. 캐시 구간이 맞지 않으면 ValueError. 가격 행이 없으면 빈 DataFrame.
    입력 타입->반환 타입: (str, str, str, dict|None, pd.DataFrame|None) -> pd.DataFrame
    """
    if precomputed_indicators is not None and precomputed_prices is not None:
        return _assemble_features(None, ticker, start_date, end_date, precomputed_indicators, precomputed_prices)
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            return _assemble_features(cur, ticker, start_date, end_date, precomputed_indicators, precomputed_prices)


def _assemble_features(cur, ticker, start_date, end_date, precomputed_indicators, precomputed_prices):
    """
    설명: get_features 본체 — 가격을 캐시 또는 DB에서 얻고 지표 9종을 _asof_join으로 붙인다.
    주의 사항: cur는 캐시가 없는 쪽(가격 또는 지표)을 읽을 때만 쓰며, 두 캐시가 다 있으면 None이어도 된다.
    가격 캐시의 span이 요청 구간과 다르거나 지표 캐시 하한이 start_date보다 늦으면 ValueError.
    입력 타입->반환 타입: (psycopg.Cursor|None, str, str, str, dict|None, pd.DataFrame|None) -> pd.DataFrame
    """
    if precomputed_prices is not None:
        requested = (str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date()))
        if precomputed_prices.attrs.get("span") != requested:
            raise ValueError(
                f"{ticker}: 가격 캐시 구간({precomputed_prices.attrs.get('span')})이 요청 구간({requested})과 다름"
            )
        prices = precomputed_prices
    else:
        prices = _load_prices(cur, ticker, start_date, end_date)
    if prices.empty:
        return prices.set_index("date")

    features = prices.set_index("date")
    trading_dates = pd.Series(features.index)

    for code in INDICATOR_CODES:
        if precomputed_indicators is not None:
            indicator_df = precomputed_indicators[code]
            cache_start = indicator_df.attrs.get("start_date")
            if cache_start is not None and pd.Timestamp(cache_start) > pd.Timestamp(start_date):
                raise ValueError(
                    f"{code}: 지표 캐시 하한({cache_start})이 요청 start_date({start_date})보다 늦음 — "
                    f"load_indicator_cache(end_date, start_date)를 같은 start_date로 다시 만들 것"
                )
        else:
            indicator_df = _load_indicator_series(cur, code, end_date, start_date)
        features[code] = _asof_join(trading_dates, indicator_df).values

    return features


def print_nan_ratio(features):
    """
    설명: 피처 행렬의 컬럼별 NaN 비율(%)을 큰 순서로 출력한다.
    주의 사항: 점검용(이 파일 __main__에서만 쓴다).
    입력 타입->반환 타입: (pd.DataFrame) -> None
    """
    print("\n=== 컬럼별 NaN 비율 ===")
    ratios = features.isna().mean().sort_values(ascending=False) * 100
    for col, pct in ratios.items():
        print(f"  {col}: {pct:.2f}%")


def verify_no_leakage(ticker, start_date, end_date, indicator_code, n=10):
    """
    설명: 지표 하나에서 거래일 n개를 표본으로 골라, 붙은 값의 published_date가 그 거래일보다 앞서는지 출력한다.
    주의 사항: 점검용(이 파일 __main__에서만 쓴다). DB 읽기. 표본은 random_state=42로 고정.
    입력 타입->반환 타입: (str, str, str, str, int) -> None
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            prices = _load_prices(cur, ticker, start_date, end_date)
            if prices.empty:
                print(f"{ticker}: 가격 데이터 없음 ({start_date}~{end_date})")
                return
            indicator_df = _load_indicator_series(cur, indicator_code, end_date)

    if indicator_df.empty:
        print(f"{indicator_code}: 지표 데이터 없음")
        return

    left = prices[["date"]].sort_values("date")
    right = indicator_df[["pub_date", "value"]].sort_values("pub_date")
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="pub_date",
        direction="backward", allow_exact_matches=False,
    ).dropna(subset=["value"])

    sample = merged.sample(min(n, len(merged)), random_state=42).sort_values("date")

    print(f"\n=== 누수 검증: {indicator_code}, 샘플 {len(sample)}건 ===")
    all_ok = True
    for _, row in sample.iterrows():
        trading_day = row["date"].date()
        pub_date = row["pub_date"].date()
        ok = pub_date < trading_day
        all_ok &= ok
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] 거래일={trading_day}  value={row['value']:.4f}  published_date={pub_date}")
    print(f"전체 통과 (published_date < 거래일): {all_ok}")


if __name__ == "__main__":
    TICKER = "005930"
    START = "2024-01-01"
    END = "2024-03-31"

    features = get_features(TICKER, START, END)

    print(f"=== get_features({TICKER}, {START}, {END}) ===")
    print(f"shape: {features.shape}")
    print(f"columns: {list(features.columns)}")

    print_nan_ratio(features)

    # 월별 지표 하나(CPI)를 골라 누수 여부 샘플 검증
    verify_no_leakage(TICKER, START, END, "ECOS_901Y009_0", n=10)
