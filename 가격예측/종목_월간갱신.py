# 종목_월간갱신.py
# Task T pooled 변동성 모델의 종목 유니버스를 월 1회 자동 갱신한다(2026-09-07 신규 작성 —
# CLAUDE.md "다음 세션 시작 시 할 일" 3번 항목).
#
# 로직: fdr.StockListing('KOSPI') 재조회 -> 우선주 제외(2026-09-06/09-07 세션과 동일 정규식,
# 시가총액 상위 TARGET_UNIVERSE_SIZE(기본 100)위 재산정 -> constants.STOCKS와 비교해:
#   - 신규 진입(상위권에 새로 들어왔고 현재 목록에 없거나 비활성) -> 가격 데이터 없으면 백필 후
#     MIN_TRAIN_SEQUENCES(가격예측_변동성_공통.py, 오늘 세션 기준 10) 이상인지 확인하고
#     통과해야만 활성화
#   - 이탈(현재 active=True인데 더 이상 상위권 아님) -> active=False로 전환
# constants.py는 절대 통째로 재생성하지 않는다 — 해당 Stock(...) 줄만 정확히 찾아 boolean만
# 바꾸거나(True<->False), STOCKS 리스트 닫는 대괄호 바로 앞에 새 Stock(...) 줄만 추가한다.
# 기존 주석·이력은 전부 그대로 보존된다(CLAUDE.md "기존 컬럼 값을 파괴적으로 덮어쓰지 않는다"
# 원칙을 소스 파일 수정에도 동일하게 적용).
#
# 안전장치: 기본은 --dry-run(제안만 출력, 파일 변경 없음)이다. 실제로 constants.py를 고치려면
# --apply를 명시해야 하고, 그때도 수정 직전 constants.py.bak_<타임스탬프>로 백업을 남긴다.
# 무인 크론에서 매달 이 스크립트가 소스 코드를 실제로 고치는 것 자체가 상당히 민감한 동작이라
# (CLAUDE.md의 "하드 투 리버스 작업은 확인 후" 원칙), 스케줄러에 등록할 때 --apply를 명시적으로
# 넣을지는 사람이 결정할 문제로 남겨둔다 — 이 세션에서는 코드만 작성하고 실행하지 않는다.
#
# 실행(다음 세션 이후): python -m 가격예측.종목_월간갱신 --dry-run  (제안만 확인)
#                        python -m 가격예측.종목_월간갱신 --apply    (실제로 constants.py 수정)

import argparse
import os
import re
import shutil
import sys
from datetime import date, datetime

import FinanceDataReader as fdr

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, STOCKS
from 가격예측.가격예측_변동성_공통 import MIN_TRAIN_SEQUENCES
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.sequence_dataset import build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset_v2
from 주가데이터.주가_공통 import _needs_initial_load, update_stock_data

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 2026-09-06/09-07 세션에서 검증된 우선주 필터 — 이름이 "...우"/"...우B"/"...2우" 등으로
# 끝나는 종목을 제외한다. 당시 실측 결과 943개 중 109개 제외로 정확히 일치했다(재확인용
# 참고 수치 — 아래 _PREFERRED_COUNT_SANITY_RANGE로 매달 큰 이탈이 있으면 경고).
_PREFERRED_PATTERN = re.compile(r".*\d?우[A-Z]?$")
_PREFERRED_COUNT_SANITY_RANGE = (60, 220)  # 109 기준으로 넉넉히 잡은 범위 — 벗어나면 정규식이
# KRX 명명 규칙 변화로 깨졌을 가능성을 의심할 것(원래는 정지가 아니라 경고만 하고 계속 진행).

TARGET_UNIVERSE_SIZE = 100  # 사람이 의도적으로 바꾸는 값 — len(ACTIVE_TICKERS)에서 자동
# 유도하지 않는다(다른 이유로 몇 종목이 수동 비활성화됐을 때 유니버스 크기가 조용히 줄어드는
# 것을 방지).

CONSTANTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "constants.py")

_STOCK_LINE_PATTERN = re.compile(r'^(\s*)Stock\("([^"]+)",\s*"([^"]*)",\s*(True|False)\)(,?)(.*)$')


# ── 1) 시가총액 상위 유니버스 재조회 ─────────────────────────────────────────────────

def fetch_top_n_universe(n=TARGET_UNIVERSE_SIZE):
    """fdr.StockListing('KOSPI')에서 우선주를 제외하고 시가총액 상위 n개를 반환한다.
    반환: list[(ticker, name, marcap)], 시가총액 내림차순."""
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
    """daily_stock_prices에 해당 종목 데이터가 전혀 없으면 초기적재한다(월간갱신이 신규
    종목을 완전히 자동으로 처리할 수 있도록 — 사람이 별도로 주가_초기적재.py를 먼저
    돌려줄 필요가 없게 함)."""
    if _needs_initial_load(ticker):
        print(f"  {ticker}: 가격 데이터 없음 — {STOCK_INITIAL_LOAD_START}부터 초기적재")
        update_stock_data(ticker, start_date_override=STOCK_INITIAL_LOAD_START)


def count_train_sequences(ticker, train_end, val_end, lookback=20):
    """가격예측_변동성_공통.MIN_TRAIN_SEQUENCES 게이팅을 위한 실제 train 시퀀스 수를 센다.

    build_merged_dataset_v2(방향용, GARCH 적합 불필요)로 계산한다 — build_merged_dataset_v2와
    build_merged_dataset_v2_volatility_hybrid는 momentum_feature.py의 WINDOW=60이 recent_vol_
    ma20(20일)·garch_sigma(사실상 1행)보다 훨씬 긴 선행 결측을 만들어 momentum 쪽이 항상
    binding constraint가 되므로, 두 데이터셋의 유효 행 수(따라서 시퀀스 수)가 항상 동일하다
    (2026-09-07 100종목 코드 리뷰 세션에서 구조적으로 검증됨 — pooled_dataset.py의
    train_end 일치 assert가 매 실행 통과하는 이유와 동일한 근거). GARCH 적합 없이 더 싸게
    같은 결과를 얻기 위해 방향용 데이터셋을 쓴다."""
    merged, _ = build_merged_dataset_v2(ticker, STOCK_INITIAL_LOAD_START, date.today().isoformat())
    feature_cols = [c for c in merged.columns if c != "target"]
    X, y, dates = build_sequences(merged, feature_cols, lookback)
    splits = split_sequences_by_date(X, y, dates, train_end, val_end)
    return len(splits["train"][0])


# ── 3) constants.py 원본 텍스트 파싱/수정 ────────────────────────────────────────────

def _read_constants_lines():
    with open(CONSTANTS_PATH, "r", encoding="utf-8") as f:
        return f.read().split("\n")


def _find_stocks_block(lines):
    """STOCKS = [ 로 시작하는 줄과, 그 뒤 첫 '](닫는 대괄호만 있는 줄)'을 찾는다."""
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
    """STOCKS 블록 안에서 실제 Stock(...) 줄만 파싱한다. 반환: {ticker: line_idx}."""
    line_idx_by_ticker = {}
    for i in range(start + 1, end):
        m = _STOCK_LINE_PATTERN.match(lines[i])
        if m:
            ticker = m.group(2)
            line_idx_by_ticker[ticker] = i
    return line_idx_by_ticker


def _set_active_flag(lines, line_idx, new_active, reason):
    """해당 줄의 True/False만 바꾸고, 줄 끝에 자동 주석을 덧붙인다(기존 주석이 있으면
    그 뒤에 이어붙임 — 지우지 않음)."""
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
    """STOCKS 리스트를 닫는 ']' 바로 앞(end_idx)에 새 Stock(...) 줄들을 삽입한다.
    new_entries: list[(ticker, name, reason)]."""
    insert_lines = []
    for ticker, name, reason in new_entries:
        comment = f"  # 자동갱신 {date.today().isoformat()}: {reason}"
        insert_lines.append(f'    Stock("{ticker}", "{name}", True),{comment}')
    return lines[:end_idx] + insert_lines + lines[end_idx:]


def apply_updates_to_constants(deactivate, reactivate, add_new):
    """deactivate: list[(ticker, reason)] — active=True -> False
       reactivate: list[(ticker, reason)] — active=False -> True (이미 목록에 있던 종목)
       add_new:    list[(ticker, name, reason)] — 완전히 새로운 종목 추가(active=True)

    수정 전 constants.py.bak_<타임스탬프>로 백업한다."""
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
    """현재 constants.STOCKS와 새로 조회한 시가총액 상위 target_size를 비교해 변경 계획을
    세운다(constants.py는 아직 건드리지 않음 — 순수 계획 수립).

    반환: dict(deactivate=[(ticker,reason)], reactivate=[(ticker,reason)],
               add_new=[(ticker,name,reason)], skipped_thin=[(ticker,name,n_seq)],
               unchanged_count=int)"""
    universe = fetch_top_n_universe(target_size)
    universe_tickers = {t for t, _, _ in universe}
    name_by_ticker = {t: name for t, name, _ in universe}

    current_by_ticker = {s.ticker: s for s in STOCKS}
    current_active = set(ACTIVE_TICKERS)

    ref, _ = build_merged_dataset_v2("005930", STOCK_INITIAL_LOAD_START, date.today().isoformat())
    train_end, val_end = compute_global_split_dates(ref.index)

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
