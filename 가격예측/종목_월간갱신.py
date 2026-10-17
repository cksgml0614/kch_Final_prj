# 종목_월간갱신.py — 시가총액 상위 종목을 다시 조회해 종목 유니버스(constants.STOCKS) 변경 계획을 세우고, --apply일 때만 반영한다.
# 파이프라인 단계: 월간 유니버스 관리(monthly_universe_check.yml, dry-run 전용). 주요 진입 함수: main.
# 실행: python -m 가격예측.종목_월간갱신 --dry-run(기본, 계획만) / --apply(constants.py 수정, 사람이 로컬에서).
# 배경: 결정기록.md의 "월간 유니버스 갱신은 기본 dry-run, constants.py는 해당 줄만 고친다"

import argparse
import os
import re
import shutil
import sys
from datetime import date, datetime

import FinanceDataReader as fdr

from constants import ACTIVE_TICKERS, MIN_TRAIN_SEQUENCES, STOCK_INITIAL_LOAD_START, STOCKS
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import valid_feature_dates
from 주가데이터.주가_공통 import needs_initial_load, update_stock_data

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 우선주 이름 패턴("...우"/"...우B"/"...2우" 등).
_PREFERRED_PATTERN = re.compile(r".*\d?우[A-Z]?$")
_PREFERRED_COUNT_SANITY_RANGE = (60, 220)  # 우선주 제외 건수가 이 범위를 벗어나면 이름 규칙 변화를 의심(경고만).

TARGET_UNIVERSE_SIZE = 100  # 사람이 정하는 값 — 비활성 종목이 생겨도 줄지 않게 ACTIVE_TICKERS에서 유도하지 않는다.

CONSTANTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "constants.py")

_STOCK_LINE_PATTERN = re.compile(r'^(\s*)Stock\("([^"]+)",\s*"([^"]*)",\s*(True|False)\)(,?)(.*)$')


# ── 1) 시가총액 상위 유니버스 재조회 ─────────────────────────────────────────────────

def fetch_top_n_universe(n=TARGET_UNIVERSE_SIZE):
    """
    설명: KOSPI 상장 목록에서 우선주를 빼고 시가총액 상위 n개를 [(ticker, name, marcap)] 내림차순으로 돌려준다.
    주의 사항: 네트워크(FinanceDataReader). 우선주 제외 건수가 예상 범위를 벗어나면 경고만 낸다.
    입력 타입->반환 타입: (int) -> list[tuple[str, str, float]]
    """
    df = fdr.StockListing("KOSPI")[["Code", "Name", "Marcap"]].copy()
    is_pref = df["Name"].apply(lambda name: bool(_PREFERRED_PATTERN.match(name)))
    n_pref = int(is_pref.sum())
    lo, hi = _PREFERRED_COUNT_SANITY_RANGE
    if not (lo <= n_pref <= hi):
        print(f"⚠️ 우선주 필터 제외 건수({n_pref})가 예상 범위({lo}~{hi})를 벗어났습니다 — "
              f"KRX 종목명 규칙이 바뀌었을 수 있습니다. 결과를 사람이 검토하세요.")

    common = df[~is_pref].sort_values("Marcap", ascending=False).reset_index(drop=True)
    top_n = common.head(n)
    return [(row["Code"], row["Name"], float(row["Marcap"])) for _, row in top_n.iterrows()]


# ── 2) 신규 진입 종목 게이팅(최소 train 시퀀스 수 확인) ────────────────────────────────

def ensure_price_data(ticker):
    """
    설명: daily_stock_prices에 종목 데이터가 전혀 없으면 STOCK_INITIAL_LOAD_START부터 초기적재한다.
    주의 사항: DB 쓰기(dry-run에서도 실행된다 — 유니버스 밖 종목 주가가 DB에 들어감).
    입력 타입->반환 타입: (str) -> None
    """
    if needs_initial_load(ticker):
        print(f"  {ticker}: 가격 데이터 없음 — {STOCK_INITIAL_LOAD_START}부터 초기적재")
        update_stock_data(ticker, start_date_override=STOCK_INITIAL_LOAD_START)


def count_train_sequences(ticker, train_end, val_end, lookback=20):
    """
    설명: GARCH σ 전 변동성 frame의 유효 날짜로 이 종목의 train 시퀀스 수(target일 <= train_end)를 센다.
    주의 사항: DB 읽기. lookback은 운영 LOOKBACK과 같아야 실제 학습 시퀀스 수와 맞는다. val_end는 쓰지 않는다.
    입력 타입->반환 타입: (str, pd.Timestamp, pd.Timestamp, int) -> int
    """
    dates = valid_feature_dates(ticker, STOCK_INITIAL_LOAD_START, date.today().isoformat())
    seq_dates = dates[lookback - 1:]
    return int((seq_dates <= train_end).sum())


# ── 3) constants.py 원본 텍스트 파싱/수정 ────────────────────────────────────────────

def _read_constants_lines():
    """
    설명: constants.py를 줄 목록으로 읽는다.
    주의 사항: 파일 읽기.
    입력 타입->반환 타입: () -> list[str]
    """
    with open(CONSTANTS_PATH, "r", encoding="utf-8") as f:
        return f.read().split("\n")


def _find_stocks_block(lines):
    """
    설명: 'STOCKS = [' 줄과 그 뒤 처음 나오는 ']'만 있는 줄의 위치를 돌려준다.
    주의 사항: 둘 중 하나라도 없으면 RuntimeError.
    입력 타입->반환 타입: (list[str]) -> (int, int)
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("STOCKS = ["):
            start = i
            break
    if start is None:
        raise RuntimeError("constants.py에서 'STOCKS = [' 를 찾지 못함 — 수동 확인 필요")
    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "]":
            end = i
            break
    if end is None:
        raise RuntimeError("constants.py에서 STOCKS 리스트를 닫는 ']' 를 찾지 못함 — 수동 확인 필요")
    return start, end


def _parse_stock_lines(lines, start, end):
    """
    설명: STOCKS 블록 안의 Stock(...) 줄을 찾아 {ticker: 줄 위치}로 돌려준다.
    주의 사항: _STOCK_LINE_PATTERN 형식이 아닌 줄은 건너뛴다.
    입력 타입->반환 타입: (list[str], int, int) -> dict[str, int]
    """
    line_idx_by_ticker = {}
    for i in range(start + 1, end):
        m = _STOCK_LINE_PATTERN.match(lines[i])
        if m:
            ticker = m.group(2)
            line_idx_by_ticker[ticker] = i
    return line_idx_by_ticker


def _set_active_flag(lines, line_idx, new_active, reason):
    """
    설명: 해당 Stock 줄의 True/False만 바꾸고 줄 끝에 자동갱신 주석을 덧붙인다(lines를 직접 수정).
    주의 사항: 기존 줄 끝 주석은 지우지 않고 그 뒤에 이어 붙인다.
    입력 타입->반환 타입: (list[str], int, bool, str) -> None
    """
    m = _STOCK_LINE_PATTERN.match(lines[line_idx])
    indent, ticker, name, _old_flag, comma, trailing = m.groups()
    new_flag = "True" if new_active else "False"
    auto_comment = f"# 자동갱신 {date.today().isoformat()}: {reason}"
    if trailing.strip():
        new_trailing = f"{trailing}  {auto_comment}"
    else:
        new_trailing = f"  {auto_comment}"
    lines[line_idx] = f'{indent}Stock("{ticker}", "{name}", {new_flag}){comma}{new_trailing}'


def _append_new_stocks(lines, end_idx, new_entries):
    """
    설명: STOCKS를 닫는 ']' 바로 앞에 새 Stock(..., True) 줄들을 넣은 새 줄 목록을 돌려준다.
    주의 사항: new_entries는 [(ticker, name, reason)].
    입력 타입->반환 타입: (list[str], int, list[tuple[str, str, str]]) -> list[str]
    """
    insert_lines = []
    for ticker, name, reason in new_entries:
        comment = f"  # 자동갱신 {date.today().isoformat()}: {reason}"
        insert_lines.append(f'    Stock("{ticker}", "{name}", True),{comment}')
    return lines[:end_idx] + insert_lines + lines[end_idx:]


def apply_updates_to_constants(deactivate, reactivate, add_new):
    """
    설명: 비활성화·재활성화·신규 추가 계획을 constants.py에 반영한다.
    주의 사항: 파일 쓰기. 먼저 constants.py.bak_<시각>으로 백업한다. 반영 뒤 constants.py를 커밋·푸시해야 Actions 수집에 반영된다.
    입력 타입->반환 타입: (list[tuple], list[tuple], list[tuple]) -> None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CONSTANTS_PATH}.bak_{timestamp}"
    shutil.copy2(CONSTANTS_PATH, backup_path)
    print(f"백업 저장: {backup_path}")

    lines = _read_constants_lines()
    start, end = _find_stocks_block(lines)
    line_idx_by_ticker = _parse_stock_lines(lines, start, end)

    for ticker, reason in deactivate:
        if ticker not in line_idx_by_ticker:
            raise RuntimeError(f"{ticker}: constants.py STOCKS에서 해당 줄을 찾지 못함 — 비활성화 불가")
        _set_active_flag(lines, line_idx_by_ticker[ticker], False, reason)

    for ticker, reason in reactivate:
        if ticker not in line_idx_by_ticker:
            raise RuntimeError(f"{ticker}: constants.py STOCKS에서 해당 줄을 찾지 못함 — 재활성화 불가")
        _set_active_flag(lines, line_idx_by_ticker[ticker], True, reason)

    if add_new:
        lines = _append_new_stocks(lines, end, add_new)

    with open(CONSTANTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"constants.py 갱신 완료 — 비활성화 {len(deactivate)}건, 재활성화 {len(reactivate)}건, "
          f"신규 추가 {len(add_new)}건")


# ── 4) 전체 갱신 로직 ────────────────────────────────────────────────────────────

def plan_universe_update(target_size=TARGET_UNIVERSE_SIZE):
    """
    설명: 현재 STOCKS와 새 시가총액 상위 target_size를 비교해 비활성화·재활성화·신규 추가·보류 계획 dict를 세운다.
    주의 사항: constants.py는 고치지 않지만 신규 후보의 주가를 DB에 초기적재한다(ensure_price_data). 분할 경계 기준은 005930.
    입력 타입->반환 타입: (int) -> dict
    """
    universe = fetch_top_n_universe(target_size)
    universe_tickers = {t for t, _, _ in universe}
    name_by_ticker = {t: name for t, name, _ in universe}

    current_by_ticker = {s.ticker: s for s in STOCKS}
    current_active = set(ACTIVE_TICKERS)

    train_end, val_end = compute_global_split_dates(
        valid_feature_dates("005930", STOCK_INITIAL_LOAD_START, date.today().isoformat()))

    deactivate = []
    for ticker in current_active:
        if ticker not in universe_tickers:
            deactivate.append((ticker, f"시가총액 상위 {target_size}위 이탈"))

    reactivate = []
    add_new = []
    skipped_thin = []
    for ticker in universe_tickers:
        if ticker in current_active:
            continue  # 이미 활성 — 변경 없음
        is_known = ticker in current_by_ticker
        name = name_by_ticker[ticker]

        print(f"신규 진입 후보: {ticker} {name} — train 시퀀스 수 확인 중...")
        ensure_price_data(ticker)
        try:
            n_seq = count_train_sequences(ticker, train_end, val_end, lookback=20)
        except Exception as e:
            print(f"  ⚠️ {ticker}: 시퀀스 계산 실패({e}) — 이번 달은 건너뜀")
            skipped_thin.append((ticker, name, -1))
            continue

        if n_seq < MIN_TRAIN_SEQUENCES:
            print(f"  ⚠️ {ticker}: train 시퀀스 {n_seq}개 < 최소 {MIN_TRAIN_SEQUENCES}개 — 활성화 보류")
            skipped_thin.append((ticker, name, n_seq))
            continue

        print(f"  ✅ {ticker}: train 시퀀스 {n_seq}개 — 활성화 가능")
        reason = f"시가총액 상위 {target_size}위 진입, train 시퀀스 {n_seq}개(최소 {MIN_TRAIN_SEQUENCES}) 통과"
        if is_known:
            reactivate.append((ticker, reason))
        else:
            add_new.append((ticker, name, reason))

    return {
        "deactivate": deactivate, "reactivate": reactivate, "add_new": add_new,
        "skipped_thin": skipped_thin,
        "unchanged_count": len(universe_tickers & current_active),
    }


def print_plan(plan):
    """
    설명: 갱신 계획(유지·비활성화·재활성화·신규 추가·보류)을 사람이 읽을 수 있게 출력한다.
    주의 사항: 없음.
    입력 타입->반환 타입: (dict) -> None
    """
    print("\n" + "=" * 100)
    print("=== 종목 유니버스 갱신 계획 ===")
    print("=" * 100)
    print(f"변경 없음(계속 활성): {plan['unchanged_count']}종목")
    print(f"\n비활성화 대상({len(plan['deactivate'])}종목):")
    for ticker, reason in plan["deactivate"]:
        print(f"  - {ticker}: {reason}")
    print(f"\n재활성화 대상({len(plan['reactivate'])}종목):")
    for ticker, reason in plan["reactivate"]:
        print(f"  + {ticker}: {reason}")
    print(f"\n신규 추가 대상({len(plan['add_new'])}종목):")
    for ticker, name, reason in plan["add_new"]:
        print(f"  + {ticker} {name}: {reason}")
    print(f"\n게이팅 미통과로 보류({len(plan['skipped_thin'])}종목):")
    for ticker, name, n_seq in plan["skipped_thin"]:
        print(f"  ~ {ticker} {name}: train 시퀀스 {n_seq}개")


def main():
    """
    설명: 인자를 읽어 갱신 계획을 세우고 출력한 뒤, --apply이고 변경이 있을 때만 constants.py에 반영한다.
    주의 사항: 계획 단계에서 DB에 쓸 수 있다(ensure_price_data). --apply는 파일을 고친다.
    입력 타입->반환 타입: () -> None
    """
    parser = argparse.ArgumentParser(
        description="Task T pooled 변동성 모델 종목 유니버스 월간 갱신 — 시가총액 상위 재조회 + 게이팅"
    )
    parser.add_argument("--universe-size", type=int, default=TARGET_UNIVERSE_SIZE,
                         help=f"목표 유니버스 크기 (기본 {TARGET_UNIVERSE_SIZE})")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True,
                        help="계획만 출력하고 constants.py는 건드리지 않음(기본값)")
    group.add_argument("--apply", action="store_true",
                        help="계획대로 constants.py를 실제로 수정(백업 자동 생성)")
    args = parser.parse_args()

    plan = plan_universe_update(target_size=args.universe_size)
    print_plan(plan)

    if not args.apply:
        print("\n(--dry-run 모드 — constants.py는 수정하지 않았습니다. 실제 적용하려면 --apply)")
        return

    if not (plan["deactivate"] or plan["reactivate"] or plan["add_new"]):
        print("\n변경 사항 없음 — constants.py를 건드리지 않습니다.")
        return

    apply_updates_to_constants(plan["deactivate"], plan["reactivate"], plan["add_new"])


if __name__ == "__main__":
    main()
