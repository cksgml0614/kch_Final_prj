# 라벨_공통.py
# Task E — CLAUDE.md D-2 절차의 z-score 계산 + 라벨 파생 함수. 라벨_생성.py(daily_labels 적재),
# 라벨_보고.py(Task E 정지 지점 보고), 감성분석/taskf_gating.py(Task F 게이팅)가 함께 참조한다.
#
# 라벨(방향 5-class/3-class, 변동성 2-class) 자체는 daily_labels에 저장하지 않는다 — z_score만
# 저장해두고 이 파일의 label_* 함수로 조회 시점에 파생한다(2026-08-30 설계 확정).
#
# ✅ 2026-08-30 사람 확정(결과_TaskE_라벨링.md 기반): 윈도우 N=20/60/120 전부 사용, 방향
# 5-class·3-class 전부 사용, 변동성 2-class는 |z|>1.0 채택. 상세 근거는 CLAUDE.md D-2 참고.
# VOLATILITY_2CLASS_THRESHOLDS(3후보)는 라벨_보고.py가 Task E 보고에서 후보 비교를 그대로
# 참조할 수 있도록 남겨뒀다 — Task F 이후 신규 코드는 CONFIRMED_VOLATILITY_THRESHOLD를 쓸 것.
#
# ✅ 2026-09-02 horizon 확장: 익일(h=1) 라벨은 노이즈 대비 신호가 약할 수 있다는 판단으로,
# h거래일(3/5/10) 누적 초과수익률 라벨을 추가했다(daily_labels에 horizon_h 컬럼 신설, PK
# 편입 — 기존 h=1 데이터는 보존). compute_z_scores()/to_records()의 옛 6-tuple(h=1 전용)
# 시그니처는 compute_multi_horizon_z_scores()/to_records()(horizon_h 포함, h=1도 그 특수
# 케이스로 처리)로 대체됐다. compute_z_scores()는 라벨_보고.py의 레짐 특성 계산(원 change_rate
# 필요, daily_labels 자체를 쓰지 않음)이 계속 참조하므로 그대로 남겨뒀다.
#
# ✅ 2026-09-06 label_basis 확장: "시장 전체에 좋은 뉴스"가 KOSPI 차감으로 상쇄되는 것 아니냐는
# 우려로, KOSPI를 빼지 않은 순수 change_rate 기준 z-score('absolute_return')를 추가했다
# (daily_labels에 label_basis 컬럼 신설, PK 편입 — 기존 행은 label_basis='excess_return'으로
# 보존). compute_multi_horizon_z_scores()/to_records()/upsert_labels()/load_labels() 모두
# basis/label_basis 인자를 받되 기본값은 'excess_return'이라 기존 호출부는 수정 없이 그대로
# 동작한다.

import pandas as pd

from constants import SPLIT_RATIOS
from db_manager import get_db_connection
from 시장지표.feature_loader import get_indicator_level_series

WINDOW_SIZES = [20, 60, 120]
HORIZONS = [1, 3, 5, 10]
LABEL_BASES = ["excess_return", "absolute_return"]

# CLAUDE.md D-2 / TASK_EF_라벨링_비교실험.md "라벨 정의"가 제시했던 임계값 후보(방향 라벨은
# 후보가 곧 확정값이었음). 변동성만 3후보 중 1.0으로 확정(위 참고).
DIRECTION_5CLASS_THRESHOLDS = (0.5, 1.5)
DIRECTION_3CLASS_THRESHOLD = 0.5
VOLATILITY_2CLASS_THRESHOLDS = [0.5, 1.0, 1.5]
CONFIRMED_VOLATILITY_THRESHOLD = 1.0


def load_stock_change_rate(cur, ticker):
    """daily_stock_prices에서 이미 계산돼 있는 change_rate를 그대로 읽는다(재계산 안 함)."""
    cur.execute(
        "SELECT date, change_rate FROM daily_stock_prices WHERE ticker = %s ORDER BY date",
        (ticker,),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "change_rate"])
    if df.empty:
        return df
    df["change_rate"] = df["change_rate"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_z_scores(ticker, cur, window_sizes=WINDOW_SIZES):
    """excess_return_t = 종목 change_rate_t - KOSPI change_rate_t,
    sigma_t(N) = 직전 N거래일 excess_return 표준편차(t 시점 미포함), z_t(N) = excess_return_t / sigma_t(N).

    종목 change_rate는 daily_stock_prices에서 직접 읽는다. KOSPI는 재수집 코드를 새로 만들지
    않고 시장지표/feature_loader.py의 get_indicator_level_series()로 원 레벨을 읽어 자체
    pct_change로 등락률을 계산한다(get_features()의 누수방지 asof 조인은 "예측 피처"용이라
    당일 자기 자신의 등락률에는 맞지 않음 — feature_loader.py 함수 docstring 참고).

    가격 이력 전체(daily_stock_prices 시작일부터)를 대상으로 계산한다 — TRAIN_PERIOD_START부터만
    계산하면 윈도우 120일 롤링에 필요한 버퍼가 부족해 학습 구간 초반이 불필요하게 NULL이 된다.

    반환: DataFrame(date, stock_change_rate, kospi_change_rate, excess_return,
                    sigma_20, z_20, sigma_60, z_60, sigma_120, z_120) — window_sizes에 따라 컬럼 수 가변.
    빈 DataFrame: 가격 데이터가 아예 없는 경우.
    """
    stock_df = load_stock_change_rate(cur, ticker)
    if stock_df.empty:
        return pd.DataFrame()

    end_date = stock_df["date"].max().date().isoformat()
    kospi_level = get_indicator_level_series("KOSPI", end_date)
    if kospi_level.empty:
        raise RuntimeError("KOSPI(market_indicators) 데이터 없음 — Task G-2 적재 상태를 확인할 것")

    kospi_level = kospi_level.sort_values("date").reset_index(drop=True)
    kospi_level["kospi_change_rate"] = kospi_level["value"].pct_change()

    merged = (
        stock_df.rename(columns={"change_rate": "stock_change_rate"})
        .merge(kospi_level[["date", "kospi_change_rate"]], on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged["excess_return"] = merged["stock_change_rate"] - merged["kospi_change_rate"]

    for n in window_sizes:
        # shift(1) 먼저 적용 -> rolling(N)이 보는 마지막 값이 t-1이 됨 (t 시점 미포함)
        shifted = merged["excess_return"].shift(1)
        merged[f"sigma_{n}"] = shifted.rolling(window=n, min_periods=n).std()
        merged[f"z_{n}"] = merged["excess_return"] / merged[f"sigma_{n}"]

    return merged


def compute_multi_horizon_z_scores(ticker, cur, window_sizes=WINDOW_SIZES, horizons=HORIZONS, basis="excess_return"):
    """h거래일 누적 수익률 기준 라벨(2026-09-02 horizon 확장, 2026-09-06 basis 확장).

    basis='excess_return'(기본, 기존): cum_t,h = t부터 h거래일(t..t+h-1) (종목-KOSPI) excess_return 합계.
    basis='absolute_return'(신규): cum_t,h = 같은 구간의 종목 순수 change_rate 합계(KOSPI 차감 없음)
    — "시장 전체에 좋은 뉴스"가 KOSPI에도 반영돼 초과수익률 계산에서 상쇄될 수 있다는 우려로 추가.
    h=1이면 각 basis의 당일 값 그 자체와 정확히 같다(기존 compute_z_scores()와 값이 완전히 일치 —
    재적재 시 IS DISTINCT FROM에 걸려 h=1/excess_return 행이 바뀌지 않는 것으로 검증됨).
    sigma_t,h = 직전 window_n개 "앵커일"의 동일 basis cum_*,h 표준편차(t 시점 미포함, D-2와 동일
    원칙 — 자기 자신의 크기로 자기 자신을 정규화하지 않기 위함).
    z_t,h = cum_t,h / sigma_t,h.

    데이터 끝부분(t+h-1이 보유한 가격 이력 범위를 넘는 날짜)은 cum_t,h 자체가 NaN이 되어 sigma/z도
    함께 NULL로 저장된다("기간 끝부분은 라벨 NULL" 요건).

    구현은 compute_z_scores()와 별개 함수로 둔다 — 이미 검증된 h=1 파이프라인(라벨_보고.py의
    레짐 특성 계산이 참조)을 건드리지 않기 위해 종목/KOSPI 로딩 부분을 의도적으로 중복시켰다.
    KOSPI는 basis='absolute_return'일 때도 그대로 로드한다 — 두 basis가 같은 거래일 캘린더
    (KOSPI와 inner join된 날짜)를 공유해야 비교가 성립하기 때문이다.

    반환: DataFrame(date, stock_change_rate, kospi_change_rate, excess_return,
                    cum_1, sigma_20_1, z_20_1, ..., cum_10, sigma_120_10, z_120_10)
    """
    stock_df = load_stock_change_rate(cur, ticker)
    if stock_df.empty:
        return pd.DataFrame()

    end_date = stock_df["date"].max().date().isoformat()
    kospi_level = get_indicator_level_series("KOSPI", end_date)
    if kospi_level.empty:
        raise RuntimeError("KOSPI(market_indicators) 데이터 없음 — Task G-2 적재 상태를 확인할 것")

    kospi_level = kospi_level.sort_values("date").reset_index(drop=True)
    kospi_level["kospi_change_rate"] = kospi_level["value"].pct_change()

    merged = (
        stock_df.rename(columns={"change_rate": "stock_change_rate"})
        .merge(kospi_level[["date", "kospi_change_rate"]], on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged["excess_return"] = merged["stock_change_rate"] - merged["kospi_change_rate"]
    base_col = {"excess_return": "excess_return", "absolute_return": "stock_change_rate"}[basis]

    for h in horizons:
        # 뒤에서부터 굴린 뒤 다시 뒤집는 표준 트릭으로 "t부터 h거래일 합"(전방 롤링 합)을 만든다.
        # 뒤에 h-1개 미만 남은 행(데이터 끝부분)은 자동으로 NaN.
        reversed_base = merged[base_col][::-1]
        merged[f"cum_{h}"] = reversed_base.rolling(window=h, min_periods=h).sum()[::-1].values

        for n in window_sizes:
            shifted = merged[f"cum_{h}"].shift(1)
            merged[f"sigma_{n}_{h}"] = shifted.rolling(window=n, min_periods=n).std()
            merged[f"z_{n}_{h}"] = merged[f"cum_{h}"] / merged[f"sigma_{n}_{h}"]

    return merged


def to_records(merged, ticker, window_sizes=WINDOW_SIZES, horizons=HORIZONS, basis="excess_return"):
    """compute_multi_horizon_z_scores() 결과를 daily_labels UPSERT용 (ticker, date, window_n,
    horizon_h, label_basis, excess_return, sigma, z_score) 튜플 리스트로 펼친다. NaN은
    None(NULL)으로 변환한다 — 초기 N일이나 데이터 끝부분(h일 뒤 데이터 없음)처럼 계산할 수 없는
    행은 excess_return(=cum_h)/sigma/z_score 각각 계산 가능한 만큼만 채우고 나머지는 NULL로
    저장한다(행 자체를 빼지 않음 — D-2 "초기 N일은 라벨 NULL"과 동일한 원칙)."""
    records = []
    for row in merged.itertuples(index=False):
        d = row.date.date()
        for h in horizons:
            cum = getattr(row, f"cum_{h}")
            cum_val = None if pd.isna(cum) else float(cum)
            for n in window_sizes:
                sigma = getattr(row, f"sigma_{n}_{h}")
                z = getattr(row, f"z_{n}_{h}")
                records.append((
                    ticker, d, n, h, basis,
                    cum_val,
                    None if pd.isna(sigma) else float(sigma),
                    None if pd.isna(z) else float(z),
                ))
    return records


def upsert_labels(cur, records):
    """records: (ticker, date, window_n, horizon_h, label_basis, excess_return, sigma, z_score)
    튜플 리스트. market_indicators(G-1 계약)와 동일한 IS DISTINCT FROM + RETURNING 패턴으로
    신규/갱신을 구분한다.
    ⚠️ 2026-09-06 label_basis 추가로 8-tuple이 됨(이전 7-tuple에서 변경) — PK도
    (ticker,date,window_n,horizon_h,label_basis)로 확장됨."""
    query = """
        INSERT INTO daily_labels (ticker, date, window_n, horizon_h, label_basis, excess_return, sigma, z_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date, window_n, horizon_h, label_basis) DO UPDATE
            SET excess_return = EXCLUDED.excess_return,
                sigma = EXCLUDED.sigma,
                z_score = EXCLUDED.z_score
            WHERE daily_labels.excess_return IS DISTINCT FROM EXCLUDED.excess_return
               OR daily_labels.sigma IS DISTINCT FROM EXCLUDED.sigma
               OR daily_labels.z_score IS DISTINCT FROM EXCLUDED.z_score
        RETURNING (xmax <> 0) AS was_update
    """
    inserted = 0
    updated = 0
    for rec in records:
        cur.execute(query, rec)
        result = cur.fetchone()
        if result is None:
            continue  # 값이 동일해 WHERE 절에 걸려 스킵된 기존 행
        if result[0]:
            updated += 1
        else:
            inserted += 1
    return inserted, updated


# ── 라벨 파생 함수 (조회 시점 전용 — DB에 저장하지 않음) ──────────────────────

def label_direction_5class(z, t1=DIRECTION_5CLASS_THRESHOLDS[0], t2=DIRECTION_5CLASS_THRESHOLDS[1]):
    if z is None or pd.isna(z):
        return None
    if z <= -t2:
        return -2
    if z <= -t1:
        return -1
    if z < t1:
        return 0
    if z < t2:
        return 1
    return 2


def label_direction_3class(z, t=DIRECTION_3CLASS_THRESHOLD):
    if z is None or pd.isna(z):
        return None
    if z <= -t:
        return -1
    if z < t:
        return 0
    return 1


def label_volatility_2class(z, t):
    """부호 없이 강도만: 1=|z|>t(고변동), 0=그 외(정상)."""
    if z is None or pd.isna(z):
        return None
    return 1 if abs(z) > t else 0


# ── daily_labels 조회 + split 유틸 (라벨_보고.py/taskf_gating.py 공용) ──────────

def load_labels(ticker, horizon_h=1, label_basis="excess_return"):
    """daily_labels 조회 전용(쓰기 없음). ticker의 특정 horizon_h(기본 1=익일, 기존 호출부
    하위 호환) x label_basis(기본 'excess_return', 기존 호출부 하위 호환) z_score 이력을
    long format(date, window_n, excess_return, sigma, z_score)으로 로드한다. horizon_h=3/5/10을
    넘기면 h거래일 누적 라벨을(2026-09-02 확장), label_basis='absolute_return'을 넘기면 KOSPI
    차감 없는 순수 change_rate 기준 라벨을 로드한다(2026-09-06 확장)."""
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — 라벨_공통.load_labels")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, window_n, excess_return, sigma, z_score
                FROM daily_labels
                WHERE ticker = %s AND horizon_h = %s AND label_basis = %s
                ORDER BY date, window_n
                """,
                (ticker, horizon_h, label_basis),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "window_n", "excess_return", "sigma", "z_score"])
    if df.empty:
        return df
    for col in ["excess_return", "sigma", "z_score"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def split_dates_by_ratio(dates, ratios=SPLIT_RATIOS):
    """date 오름차순 유니크 거래일을 셔플 없이 순서대로 분할 (kobert_dataset.split_by_ratio와
    동일 원칙). constants.SPLIT_RATIOS 사용 — TASK_EF "학습 기간 하드코딩 금지".
    Task E 보고(라벨_보고.py)와 Task F 게이팅(taskf_gating.py)이 정확히 같은 날짜 경계를 쓰도록
    이 함수 하나로 통일한다 — 두 곳에서 각자 계산하면 미세한 구현 차이로 split 경계가 어긋날
    수 있다."""
    dates = pd.to_datetime(pd.Series(sorted(dates)))
    n = len(dates)
    train_ratio, val_ratio, _ = ratios
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return dates.iloc[:train_end], dates.iloc[train_end:val_end], dates.iloc[val_end:]
