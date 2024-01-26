@echo off

:: Activate the Conda environment
call conda activate ds-env

:: Run the mlflow server command
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234

:: Open the URL in the default browser
start http://localhost:1234