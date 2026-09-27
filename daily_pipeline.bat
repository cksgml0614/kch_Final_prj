@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

rem ENCODING NOTE: this file is saved as CP949 (ANSI, Korean OEM code page), NOT UTF-8.
rem Do NOT add "chcp 65001" and do NOT re-save as UTF-8: after chcp 65001, cmd.exe
rem miscomputes file offsets on lines containing multibyte UTF-8 text and starts
rem executing fragments of later lines as commands (reproduced 2026-09-24 in a fresh
rem cmd window). Keep all non-ASCII text limited to the module name below.
rem PYTHONUTF8=1 makes Python write UTF-8 into the (redirected) log file regardless
rem of the console code page, so emoji/Korean in script output do not crash.
set PYTHONUTF8=1

rem Price/indicator collection is done in the cloud (GitHub Actions,
rem daily_data_collection.yml), so this local pipeline only trains/predicts.

call .venv\Scripts\activate.bat

set LOGDIR=logs
if not exist %LOGDIR% mkdir %LOGDIR%
set LOGFILE=%LOGDIR%\daily_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.log
set LOGFILE=%LOGFILE: =0%

echo [%date% %time%] daily pipeline start >> %LOGFILE%

rem MLflow self-start: do not depend on the separate "MLflow Server" (ONLOGON) task.
rem If nothing answers on port 5000, launch mlflow_server_start.bat in the background
rem and wait (max ~60s) until /health responds. "ping" is used as the sleep because
rem "timeout" fails when stdin is not a console (Task Scheduler).
set MLFLOW_URL=http://127.0.0.1:5000/health
curl -s -o NUL --max-time 3 %MLFLOW_URL%
if errorlevel 1 (
    echo [mlflow] not running on port 5000 - starting mlflow_server_start.bat >> %LOGFILE%
    start "MLflow Server" /min cmd /c "mlflow_server_start.bat >> %LOGDIR%\mlflow_server.log 2>&1"
    set MLFLOW_UP=0
    for /l %%i in (1,1,30) do (
        if !MLFLOW_UP!==0 (
            ping -n 3 127.0.0.1 > NUL
            curl -s -o NUL --max-time 3 %MLFLOW_URL% && set MLFLOW_UP=1
        )
    )
    if !MLFLOW_UP!==1 (
        echo [mlflow] server is up >> %LOGFILE%
    ) else (
        echo [mlflow] WARNING: server did not respond within ~60s - continuing without it >> %LOGFILE%
    )
) else (
    echo [mlflow] already running on port 5000 >> %LOGFILE%
)

echo [predict] volatility predict start >> %LOGFILE%
python -m 가격예측.가격예측_변동성_일일수집 >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [predict] volatility predict FAILED [%date% %time%] >> %LOGFILE%
    exit /b 1
)
echo [predict] volatility predict done >> %LOGFILE%

echo [%date% %time%] daily pipeline all done >> %LOGFILE%
exit /b 0
