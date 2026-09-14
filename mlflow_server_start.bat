@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000
