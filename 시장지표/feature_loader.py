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
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    # NUMERIC -> psycopg가 Decimal로 반환 -> pandas 연산에서 Decimal-float 충돌 (G-2에서 발견된 버그 재발 방지)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_indicator_series(cur, indicator_code, end_date):
    """지표 1건의 전체 히스토리(하한 없이 end_date까지)를 published_date 오름차순으로 로드.
    DB에서 값을 읽어온 직후 float으로 캐스팅한다 (G-2와 동일한 패턴)."""
    cur.execute(
        """
        SELECT date AS ref_date, value, published_date AS pub_date
        FROM market_indicators
        WHERE indicator_code = %s AND published_date <= %s
        ORDER BY published_date
        """,
        (indicator_code, end_date),
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


def get_features(ticker, start_date, end_date):
    """미래 정보 누수 없이 종목 OHLCV + 시장지표 9종을 결합한 피처 행렬을 반환한다.

    인덱스: 거래일 (daily_stock_prices 기준)
    컬럼: open, high, low, close, volume + INDICATOR_CODES 9개
    각 지표는 published_date < 거래일인 값 중 최신값 (동일자 공표분 제외).
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            prices = _load_prices(cur, ticker, start_date, end_date)
            if prices.empty:
                return prices.set_index("date")

            features = prices.set_index("date")
            trading_dates = pd.Series(features.index)

            for code in INDICATOR_CODES:
                indicator_df = _load_indicator_series(cur, code, end_date)
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
