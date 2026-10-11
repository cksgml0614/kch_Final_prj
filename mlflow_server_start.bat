@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
rem Bind to loopback only (2026-10-11). The daily pipeline and dashboard run on this PC.
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 127.0.0.1 --port 5000
