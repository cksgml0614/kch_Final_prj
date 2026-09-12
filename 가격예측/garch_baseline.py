# garch_baseline.py — Task T 변동성 예측 학계 표준 baseline (2026-09-06)
#
# 지금까지 방향 예측(상승/하락)은 TF-IDF/KoBERT/KR-FinBERT/단일종목·pooled Transformer
# 전부 무작위 수준으로 실패했다. 계량경제학은 애초에 "가격 방향은 예측 불가능(효율적
# 시장가설)"을 전제하고, 변동성만 따로 모델링하는 GARCH 계열을 표준으로 쓴다. Transformer를
# 변동성 예측에 투입하기 전에 이 학계 표준부터 baseline으로 세운다 — 이번 스크립트는 그
# baseline 자체의 성능(SMA 대비)만 확인하는 것이 목적이고, Transformer와의 비교는 범위 밖이다.
#
# 설계 결정 (요청사항 판단):
#   - 수익률 정의: **로그수익률**(log(close_t/close_(t-1)) x 100, 퍼센트 단위) 채택. GARCH
#     문헌(Engle 1982, Bollerslev 1986)의 표준 관례이자 시간 가산성이 있어 변동성 모델링에
#     적합하다. arch 패키지도 매우 작은 값(예: 0.01)에서 수렴 경고가 나는 경우가 많아 100을
#     곱해 스케일을 맞추는 것이 패키지 공식 권장 사항이다. 프로젝트의 daily_stock_prices.
#     change_rate(단순 등락률, Transformer 트랙이 쓰는 값)와는 다른 값이다 — 두 트랙이 서로
#     다른 수익률 정의를 쓰는 것은 의도된 것: Transformer 쪽은 이미 확정된 설계(pct_change)를
#     바꾸지 않고, GARCH는 그 자체 학계 관례를 따른다.
#   - 실현 변동성(타겟): |수익률|(수익률 제곱 아님) 채택 — SMA baseline이 "직전 20일 평균
#     |수익률|"로 요청됐고, GARCH가 예측하는 것도 조건부 분산이 아니라 조건부 "표준편차"(σ,
#     분산의 제곱근)이므로, 두 baseline을 같은 척도(σ ~ |r|, 제곱 아님)에서 비교해야
#     내적으로 일관된다.
#   - 종목별 독립 적합(pooling 안 함) — 파라미터가 3개뿐이라 데이터 부족 문제가 없고, 종목마다
#     변동성 구조(persistence 등)가 다르다는 것이 GARCH 채택의 전제이기도 하다.
#   - 1-step-ahead 예측의 leak-free 보장: train 구간(<=train_end)에서만 파라미터(ω, α, β)를
#     추정한 뒤, 그 값을 고정(arch의 fix())하고 train+val+test 전체 실제 수익률 이력 위에서
#     재귀적으로 조건부 분산을 계산한다(재추정 없음). test 구간 값만 추출해 평가한다 —
#     각 시점의 예측은 그 이전까지의 실제 값만 사용하므로 미래 정보 누수가 없다.
#   - split 경계는 가격예측_통합모델.py/pooled_dataset.py와 동일한 전역 경계(전 종목 캘린더
#     동일함을 이미 실측 확인)를 그대로 재사용해 Transformer 실험과 조건을 맞췄다.
#
# 2026-09-12(가격예측/ 활성·검증 재분리): 운영 파이프라인이 쓰지 않는 fit_and_forecast_garch()/
# compute_full_period_sigma()/main()은 검증용/garch_baseline_check.py로 분리했다(값 변경 없음).
# 이 파일에는 활성 파이프라인(가격예측_변동성_공통.py)이 실제로 쓰는 함수만 남는다.

import sys

import numpy as np
import pandas as pd
from arch import arch_model

from db_manager import get_db_connection

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SMA_WINDOW = 20


def load_close_prices(ticker, start_date, end_date):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — load_close_prices")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, close FROM daily_stock_prices
                WHERE ticker = %s AND date BETWEEN %s AND %s
                ORDER BY date
                """,
                (ticker, start_date, end_date),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def compute_log_returns_pct(close):
    """로그수익률 x 100(퍼센트 단위) — arch 패키지 권장 스케일. 첫 행은 NaN이라 제거."""
    log_ret = np.log(close / close.shift(1)) * 100
    return log_ret.dropna()


def rmse_mae(pred, actual):
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    return rmse, mae


def forecast_with_fixed_params(returns, params, start_pos):
    """이미 추정된(또는 캐시에서 불러온) GARCH(1,1) 파라미터를 고정한 채, returns 전체 이력
    위에서 재귀적으로 1-step-ahead 조건부 표준편차를 계산한다(2026-09-06,
    가격예측_변동성_일일수집.py의 월 1회 재추정 캐싱을 지원하기 위해 fit_and_forecast_garch에서
    분리 — 파라미터 재추정 없이 이미 가진 params로만 재귀 계산이 필요한 경우 이 함수를 직접
    쓴다).

    ⚠️ 2026-09-06 라벨링 버그 수정(중대): arch의 `forecast(start=k, horizon=1)`이 반환하는
    행 i는 "returns.index[k+i]까지의 실제 수익률(자기 자신 포함)로 계산한, returns.index[k+i+1]
    시점에 대한 예측"이다 — 즉 그 날짜 자신의 수익률이 이미 재료로 들어간 값이다(수작업 GARCH
    재귀식 sigma2[t]=omega+alpha*eps[t-1]^2+beta*sigma2[t-1]과 실측 대조로 확인:
    함수 출력[i]가 수작업 재귀식의 (k+i+1) 위치 값과 정확히 일치하고 (k+i) 위치 값과는
    불일치함). 이전 코드는 이 행을 그대로 index[k+i]에 라벨링해 "t 시점 예측이 t 자신의
    수익률을 사용"하는 명백한 미래 정보 누수였다 — 이 버그가 하이브리드 Transformer의
    garch_sigma 피처, GARCH baseline 자체의 test 평가, 조기경보 검증(`결과_변동성_조기경보_
    검증.md`)에 전부 영향을 미쳤다(상세 재검증·수정 결과는 해당 문서 갱신판 참고).

    수정: 원시 출력 행 i를 실제 의미에 맞는 날짜 index[start_pos+i+1]로 재라벨링한다. 배열
    밖(=아직 실현되지 않은 미래 1스텝)에 해당하는 마지막 값은 여기서는 버린다 — 그 값이
    필요하면(서빙 시 "내일" 예측) forecast_next_day_sigma()를 쓴다. 즉 이 함수는 이제
    start_pos+1부터 시작하는 leak-free 과거 구간만 반환한다.

    반환: sigma_pred(pandas Series, index=날짜, start_pos+1 시점부터)"""
    am_full = arch_model(returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res_fixed = am_full.fix(params)
    fc = res_fixed.forecast(horizon=1, start=start_pos, reindex=False)
    sigma_raw = np.sqrt(fc.variance.values[:, 0])

    n = len(sigma_raw)
    target_positions = np.arange(start_pos + 1, start_pos + 1 + n)
    keep = target_positions < len(returns)
    dates = returns.index[target_positions[keep]]
    return pd.Series(sigma_raw[keep], index=dates)


def forecast_next_day_sigma(returns, params):
    """서빙(추론) 전용 — returns 마지막 날짜 다음 거래일의 GARCH 1-step-ahead σ를 예측한다.
    forecast(horizon=1)을 start 없이 호출하면 내부적으로 마지막 유효 위치(len(returns)-1)를
    기준으로 예측하는데, forecast_with_fixed_params()에서 재확인한 라벨링 규칙에 따르면 이
    위치의 원시 출력은 정확히 "returns 전체(마지막 날짜 포함)를 다 써서 그 다음 날을 예측한
    값"이다 — 그래서 이 함수는 (forecast_with_fixed_params와 달리) 재라벨링이 필요 없다.
    이 함수 자체는 2026-09-06 라벨링 버그 수정 이전부터 의미상 옳았다 — 버그는
    forecast_with_fixed_params()가 과거 구간에 잘못된 날짜를 붙이던 부분에만 있었다.

    반환: float(다음 거래일 σ 예측치)"""
    am_full = arch_model(returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res_fixed = am_full.fix(params)
    fc = res_fixed.forecast(horizon=1)
    return float(np.sqrt(fc.variance.values[-1, 0]))


def compute_sma_baseline(returns, window=SMA_WINDOW):
    """직전 window일 평균 |수익률| — shift(1) 먼저 적용해 당일 값을 포함하지 않는다(leak-free,
    D-2 원칙과 동일하게 sigma 계산 시 자기 자신 미포함)."""
    return returns.abs().shift(1).rolling(window, min_periods=window).mean()
