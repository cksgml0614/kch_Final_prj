@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

chcp 65001 >nul
set PYTHONUTF8=1
rem ↑ 이 두 줄은 파일의 다른 어떤 실행 줄보다도 먼저 와야 한다(2026-09-24 재점검 후 순서
rem 변경 — 원래는 아래 한글 설명 rem이 이 두 줄보다 앞에 있었는데, cmd.exe가 배치 파일을
rem chcp 적용 전 기본 코드페이지로 읽는 실행 환경에서는 그 한글 rem 자체가 깨져 엉뚱한
rem 명령으로 오인될 이론적 위험이 있다 — chcp/PYTHONUTF8을 최우선으로 옮겨 그 위험을 없앴다).
rem
rem UTF-8 콘솔 코드페이지 + Python UTF-8 모드 — 세 스크립트 모두 한글/이모지(⚠️❌🔄🚀 등)를
rem stdout에 찍는데, 기본 cp949 콘솔에서는 UnicodeEncodeError로 그 자리에서 죽는다(이번
rem 세션에서 python -m 주가데이터.주가_일일수집을 chcp/PYTHONUTF8 없이 직접 실행했을 때
rem 실측 확인된 문제). 둘 다 걸어 이중으로 방어한다.

call .venv\Scripts\activate.bat

set LOGDIR=logs
if not exist %LOGDIR% mkdir %LOGDIR%
set LOGFILE=%LOGDIR%\daily_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.log
set LOGFILE=%LOGFILE: =0%

echo [%date% %time%] 일일 파이프라인 시작 >> %LOGFILE%

echo [1/3] 주가_일일수집 시작 >> %LOGFILE%
python -m 주가데이터.주가_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [1/3] 주가_일일수집 실패 - 중단 [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [1/3] 주가_일일수집 완료 >> %LOGFILE%

echo [2/3] 지표_일일수집 시작 >> %LOGFILE%
python -m 시장지표.지표_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [2/3] 지표_일일수집 실패 - 중단 [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [2/3] 지표_일일수집 완료 >> %LOGFILE%

echo [3/3] 가격예측_변동성_일일수집 시작 >> %LOGFILE%
python -m 가격예측.가격예측_변동성_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [3/3] 가격예측_변동성_일일수집 실패 [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [3/3] 가격예측_변동성_일일수집 완료 >> %LOGFILE%

echo [%date% %time%] 일일 파이프라인 전체 완료 >> %LOGFILE%
exit /b 0
