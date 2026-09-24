@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

rem ENCODING NOTE: this file is saved as CP949 (ANSI, Korean OEM code page), NOT UTF-8.
rem Do NOT add "chcp 65001" and do NOT re-save as UTF-8: after chcp 65001, cmd.exe
rem miscomputes file offsets on lines containing multibyte UTF-8 text and starts
rem executing fragments of later lines as commands (reproduced 2026-09-24 in a fresh
rem cmd window). Keep all non-ASCII text limited to the three module names below.
rem PYTHONUTF8=1 makes Python write UTF-8 into the (redirected) log file regardless
rem of the console code page, so emoji/Korean in script output do not crash.
set PYTHONUTF8=1

call .venv\Scripts\activate.bat

set LOGDIR=logs
if not exist %LOGDIR% mkdir %LOGDIR%
set LOGFILE=%LOGDIR%\daily_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.log
set LOGFILE=%LOGFILE: =0%

echo [%date% %time%] daily pipeline start >> %LOGFILE%

echo [1/3] price daily collect start >> %LOGFILE%
python -m 주가데이터.주가_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [1/3] price daily collect FAILED - stop [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [1/3] price daily collect done >> %LOGFILE%

echo [2/3] indicator daily collect start >> %LOGFILE%
python -m 시장지표.지표_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [2/3] indicator daily collect FAILED - stop [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [2/3] indicator daily collect done >> %LOGFILE%

echo [3/3] volatility predict start >> %LOGFILE%
python -m 가격예측.가격예측_변동성_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [3/3] volatility predict FAILED [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [3/3] volatility predict done >> %LOGFILE%

echo [%date% %time%] daily pipeline all done >> %LOGFILE%
exit /b 0
