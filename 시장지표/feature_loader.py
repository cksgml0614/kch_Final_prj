# feature_loader.py — Task G-4
# 종목 OHLCV + market_indicators 9종(KOSPI/KOSDAQ/USD_KRW + ECOS 거시지표 6종)을
# 미래 정보 누수 없이 결합한 피처 행렬을 반환한다.

import sys

import pandas as pd

from db_manager import get_db_connection

# Windows 콘솔 기본 코드페이지(cp949)는 이모지/특수문자를 인코딩하지 못해 죽는다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# G-2(FDR 3종) + G-3(ECOS 6종, 반도체 수출금액지수 제외 확정)
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
    """(date, open, high, low, close, volume) 행 목록 -> DataFrame. _load_prices와
    load_price_cache가 같은 변환을 쓰도록 분리했다(2026-10-05)."""
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    # NUMERIC -> psycopg가 Decimal로 반환 -> pandas 연산에서 Decimal-float 충돌 (G-2에서 발견된 버그 재발 방지)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_price_cache(tickers, start_date, end_date):
    """여러 종목의 daily_stock_prices를 쿼리 한 번(ticker = ANY)으로 읽어 {ticker: DataFrame}으로
    반환한다(2026-10-05, Neon 전송량 절감 — 예전에는 같은 종목·같은 구간을 get_features,
    load_close_prices, load_high_low가 각자 다시 읽었다). 각 DataFrame은 _load_prices와 같은
    스키마·변환이고, attrs에 (start_date, end_date)를 기록해 get_features()가 다른 구간 요청에
    잘못 쓰지 않도록 한다. 데이터가 없는 종목은 빈 DataFrame으로 넣는다."""
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
    """지표 1건의 히스토리(end_date까지)를 published_date 오름차순으로 로드.
    DB에서 값을 읽어온 직후 float으로 캐스팅한다 (G-2와 동일한 패턴).

    start_date(선택, 2026-10-05, Neon 전송량 절감): 주면 "published_date < start_date인 행 중
    가장 늦은 published_date" 이상만 읽는다. _asof_join은 거래일 D(>= start_date)마다
    published_date < D인 마지막 행만 쓰므로, 그보다 앞선 행은 결과에 영향이 없다 — 지표별 공표
    간격을 가정하지 않는 정확한 하한이다(시작일 이전 행이 없으면 하한 없이 전부 읽는다).
    None이면 하한 없이 전체 히스토리(기존 동작, get_indicator_level_series 등)."""
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
    df["value"] = df["value"].astype(float)  # NUMERIC -> Decimal 방지, 가장 먼저 캐스팅
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["ref_date"] = pd.to_datetime(df["ref_date"])
    return df.sort_values("pub_date")


def _asof_join(trading_dates, indicator_df):
    """예측 대상일 D에 대해 published_date < D인 값 중 가장 최근 값을 붙인다 (같은 날 공표분 제외).
    published_date 기준 asof 병합이므로 월별 지표의 forward-fill이 자연스럽게 성립하고,
    첫 공표 이전 구간은 매치가 없어 NaN으로 남는다."""
    if indicator_df.empty:
        return pd.Series([float("nan")] * len(trading_dates), index=trading_dates)

    left = pd.DataFrame({"date": trading_dates}).sort_values("date")
    right = indicator_df[["pub_date", "value"]].sort_values("pub_date")
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="pub_date",
        direction="backward", allow_exact_matches=False,
    )
    return merged.set_index("date")["value"].reindex(trading_dates)


def get_indicator_level_series(indicator_code, end_date):
    """지표의 원 레벨(level) 시계열을 date(기준일) 오름차순으로 반환한다.

    get_features()의 asof 조인은 "예측 시점 D에서 D 이전에 이미 공표된 값"만 붙이는 누수 방지
    로직이라, 일별 지표(published_date == date)라도 D 자신의 값이 아니라 D-1 값이 매칭된다.
    이 함수는 그 조인을 거치지 않고 지표 자체의 날짜별 원값이 필요할 때 쓴다 — 예: Task E가
    KOSPI 자체의 일별 등락률(day D의 종가 대비 D의 변화량, 예측 피처가 아니라 이미 확정된
    사실)을 계산할 때. 새 DB 조회를 만들지 않고 기존 _load_indicator_series를 재사용한다.

    반환: DataFrame(columns=[date, value]), date 오름차순.
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            df = _load_indicator_series(cur, indicator_code, end_date)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    return (
        df[["ref_date", "value"]]
        .rename(columns={"ref_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_indicator_cache(end_date, start_date=None):
    """INDICATOR_CODES 9종 전체의 원자료(_load_indicator_series 결과)를 한 번에 조회해
    {code: DataFrame} 딕셔너리로 반환한다(2026-09-20, 100종목 pooled 파이프라인 성능
    최적화용 — 거시지표는 종목과 무관해 매 종목 재조회가 낭비였다).

    ⚠️ 이 캐시는 end_date에 종속적이다(published_date <= end_date 필터). get_features()에
    precomputed_indicators로 넘길 때는 반드시 동일한 end_date로 호출해야 한다 — 여기서
    end_date 일치 여부를 검증하지 않으므로 호출부(pooled_dataset.py 등, 100종목이 전부 같은
    start_date/end_date를 쓰는 경우)가 책임진다.

    start_date(선택, 2026-10-05): _load_indicator_series의 하한. 주면 각 DataFrame의
    attrs["start_date"]에 기록하고, get_features()가 이보다 이른 start_date로 이 캐시를 쓰려
    하면 예외를 낸다(하한 밖 구간이 조용히 NaN이 되는 것 방지). None이면 기존과 동일.

    반환: {indicator_code: DataFrame(_load_indicator_series와 동일 스키마)}"""
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
    """미래 정보 누수 없이 종목 OHLCV + 시장지표 9종을 결합한 피처 행렬을 반환한다.

    인덱스: 거래일 (daily_stock_prices 기준)
    컬럼: open, high, low, close, volume + INDICATOR_CODES 9개
    각 지표는 published_date < 거래일인 값 중 최신값 (동일자 공표분 제외).

    precomputed_indicators(선택, 2026-09-20 추가): load_indicator_cache(end_date)의 반환값을
    그대로 넘기면 지표 9종의 DB 조회를 생략하고 캐시를 재사용한다 — 조인 로직(_asof_join,
    종목별 trading_dates 기준)은 동일하게 그대로 수행되므로 결과는 완전히 동일하다. None이면
    기존과 똑같이 매번 새로 조회한다(하위 호환, 기본값).

    precomputed_prices(선택, 2026-10-05): load_price_cache(tickers, start_date, end_date)[ticker].
    구간(attrs["span"])이 이 호출의 start_date/end_date와 다르면 예외를 낸다. 두 캐시를 모두
    넘기면 DB에 접속하지 않는다."""
    if precomputed_indicators is not None and precomputed_prices is not None:
        return _assemble_features(None, ticker, start_date, end_date, precomputed_indicators, precomputed_prices)
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            return _assemble_features(cur, ticker, start_date, end_date, precomputed_indicators, precomputed_prices)


def _assemble_features(cur, ticker, start_date, end_date, precomputed_indicators, precomputed_prices):
    """get_features 본체. cur는 캐시가 없는 쪽(가격 또는 지표)을 조회할 때만 쓴다."""
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
    print("\n=== 컬럼별 NaN 비율 ===")
    ratios = features.isna().mean().sort_values(ascending=False) * 100
    for col, pct in ratios.items():
        print(f"  {col}: {pct:.2f}%")


def verify_no_leakage(ticker, start_date, end_date, indicator_code, n=10):
    """검증용: 지표 하나를 골라, 각 거래일에 조인된 값의 published_date가
    실제로 해당 거래일보다 이전인지 샘플로 확인해 출력한다."""
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
