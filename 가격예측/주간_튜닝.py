# 주간_튜닝.py — 주 1회 하이퍼파라미터 튜닝의 실행 진입점(CLI 인자 파싱과 절전 방지만, 로직은 하이퍼파라미터_튜닝.py).
# 파이프라인 단계: 튜닝 진입점(주간_튜닝.bat이 실행). 주요 진입 함수: main.
# 운영 기본값: 결과 파일 가격예측/checkpoints/tuning/, MLflow 실험 "주간_튜닝", GARCH 캐시 읽기 전용.
# 실행: python -m 가격예측.주간_튜닝 [--end YYYY-MM-DD] [--trials 12] [--out-dir 경로] [--tracking-uri URI]

import argparse
import contextlib
import sys
from datetime import date

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측 import 하이퍼파라미터_튜닝 as tuning

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake():
    """
    설명: 블록이 도는 동안 Windows 절전 진입을 막고(SetThreadExecutionState), 끝나면 반드시 해제한다.
    주의 사항: Windows가 아니면 아무것도 하지 않는다. 화면 꺼짐은 막지 않는다(ES_DISPLAY_REQUIRED 없음).
    입력 타입->반환 타입: () -> ContextManager[None]
    """
    kernel32 = None
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[TUNE] 절전 방지 켬")
    try:
        yield
    finally:
        if kernel32 is not None:
            kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("[TUNE] 절전 방지 해제")


def main():
    """
    설명: 인자를 읽어 절전 방지 상태에서 run_tuning을 실행한다.
    주의 사항: 결과 파일과 MLflow(기본 운영 서버, 실험 "주간_튜닝")에 쓴다. DB는 읽기만. --end 생략 시 오늘.
    입력 타입->반환 타입: () -> None
    """
    parser = argparse.ArgumentParser(description="주 1회 하이퍼파라미터 튜닝 — 채택된 값만 일일 학습이 읽는다")
    parser.add_argument("--end", type=str, default=None, help="학습에 쓸 마지막 날짜 YYYY-MM-DD (생략 시 오늘)")
    parser.add_argument("--start", type=str, default=STOCK_INITIAL_LOAD_START, help="학습 시작일")
    parser.add_argument("--trials", type=int, default=tuning.N_TRIALS, help="시도 수(0번 = 기본 설정 포함)")
    parser.add_argument("--out-dir", type=str, default=None, help="결과 폴더(기본 가격예측/checkpoints/tuning)")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI(기본 운영 서버)")
    parser.add_argument("--garch-cache-dir", type=str, default=None, help="GARCH 캐시 폴더(기본 운영 캐시, 읽기 전용)")
    args = parser.parse_args()
    end_date = args.end or date.today().isoformat()
    if args.trials < 2:
        parser.error("--trials는 2 이상이어야 한다(0번 기본 설정 + 후보 1개 이상)")
    with keep_awake():
        tuning.run_tuning(ACTIVE_TICKERS, args.start, end_date, n_trials=args.trials, out_dir=args.out_dir,
                          tracking_uri=args.tracking_uri, garch_cache_dir=args.garch_cache_dir, garch_read_only=True)


if __name__ == "__main__":
    main()
