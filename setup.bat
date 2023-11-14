@echo off

rem Check for Poetry and install if not present
where poetry >nul 2>nul
if %errorlevel% neq 0 (
    echo Poetry not found, installing...
    powershell -Command "(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -"
) else (
    echo Poetry is already installed.
)

rem Initialize the environment
poetry install
