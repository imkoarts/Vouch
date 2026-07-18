@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "BUILD_VENV=.venv-build"
set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.13"
  if not defined PY_CMD py -3.12 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.12"
)
if not defined PY_CMD (
  where python >nul 2>nul || goto no_python
  set "PY_CMD=python"
)

if not exist "%BUILD_VENV%\Scripts\python.exe" (
  %PY_CMD% -m venv "%BUILD_VENV%" || goto failed
)

"%BUILD_VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip || goto failed
"%BUILD_VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.lock -r requirements-desktop.lock -r requirements-exe-build.lock || goto failed
"%BUILD_VENV%\Scripts\python.exe" -m PyInstaller --noconfirm --clean Vouch.spec || goto failed

echo.
echo Build complete:
echo   dist\Vouch\Vouch.exe
echo.
echo Place a configured .env file beside Vouch.exe for live mode.
exit /b 0

:no_python
echo Python 3.12 or 3.13 was not found.
exit /b 1

:failed
echo.
echo The executable build failed.
exit /b 1
