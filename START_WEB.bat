@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "EXIT_CODE=1"

where py >nul 2>nul
if errorlevel 1 goto try_python
py -3.13 -c "import sys" >nul 2>nul
if not errorlevel 1 goto run_py313
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 goto run_py312

:try_python
where python >nul 2>nul
if errorlevel 1 goto no_python
python launcher.py
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_py313
py -3.13 launcher.py
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_py312
py -3.12 launcher.py
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:no_python
echo Python 3.12 or 3.13 was not found.
set "EXIT_CODE=1"

:finish
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
