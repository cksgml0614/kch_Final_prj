@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

rem Weekly hyperparameter tuning (Sunday 05:30, after the 05:00 daily run).
rem ENCODING NOTE: saved as CP949 like daily_pipeline.bat. Do NOT add "chcp 65001" and do NOT
rem re-save as UTF-8 (cmd.exe misreads multibyte UTF-8 lines). Non-ASCII text is limited to the
rem module name below. PYTHONUTF8=1 keeps the Python output in the log as UTF-8.
rem The tuning writes only the price package's checkpoints\tuning folder (result file) and the MLflow experiment
rem for weekly tuning. It reads the DB, never writes it. The daily run reads the adopted values
rem the next morning; if tuning fails or is skipped, the daily run keeps using the previous values.
set PYTHONUTF8=1

call .venv\Scripts\activate.bat

set LOGDIR=logs
if not exist %LOGDIR% mkdir %LOGDIR%
set LOGFILE=%LOGDIR%\weekly_tuning_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.log
set LOGFILE=%LOGFILE: =0%

echo [%date% %time%] weekly tuning start >> %LOGFILE%

rem Daily run first: if the "Daily Pipeline" task is Running, or any cmd.exe is running
rem daily_pipeline.bat (manual runs too), wait in 1-minute steps for at most 30 minutes, then give up.
rem Check exit codes: 0 = not running, 2 = running, 3 = task state unreadable (continue with a warning).
rem "ping" is the sleep because "timeout" fails when stdin is not a console (Task Scheduler).
set WAITED=0
:wait_daily
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $s = (Get-ScheduledTask -TaskName 'Daily Pipeline' -ErrorAction Stop).State } catch { $s = 'Unknown' }; $p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*daily_pipeline.bat*' }).Count; if ($s -eq 'Running' -or $p -gt 0) { exit 2 } elseif ($s -eq 'Unknown') { exit 3 } else { exit 0 }"
set CHK=!errorlevel!
if "!CHK!"=="2" (
    if !WAITED! GEQ 30 (
        echo [wait] daily pipeline still running after 30 min - giving up, no tuning this week [%date% %time%] >> %LOGFILE%
        exit /b 2
    )
    if !WAITED!==0 echo [wait] daily pipeline is running - waiting up to 30 min [%date% %time%] >> %LOGFILE%
    ping -n 61 127.0.0.1 > NUL
    set /a WAITED+=1
    goto wait_daily
)
if "!CHK!"=="3" echo [wait] WARNING: could not read the "Daily Pipeline" task state - continuing >> %LOGFILE%
if !WAITED! GTR 0 echo [wait] daily pipeline finished after about !WAITED! min - continuing >> %LOGFILE%

rem MLflow self-start (same as daily_pipeline.bat). Without the server the tuning still writes its
rem result file and logs [MLFLOW-SKIP].
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
        echo [mlflow] WARNING: server did not respond within ~2 min - continuing without it >> %LOGFILE%
    )
) else (
    echo [mlflow] already running on port 5000 >> %LOGFILE%
)

echo [tune] weekly tuning start >> %LOGFILE%
python -m °¡°Ý¿¹Ãø.ÁÖ°£_Æ©´× >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [tune] weekly tuning FAILED [%date% %time%] - the daily run keeps the previous values >> %LOGFILE%
    exit /b 1
)
echo [tune] weekly tuning done [%date% %time%] >> %LOGFILE%
exit /b 0
