@echo off
setlocal ENABLEDELAYEDEXPANSION

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "GRADIO_APP=%ROOT%\run\gradio_app.py"
set "REQUIREMENTS_TXT=%ROOT%\requirements.txt"

set "VENV_DIR=%ROOT%\.venv"

if not exist "%GRADIO_APP%" (
    echo [error] Could not locate run\gradio_app.py. Please ensure the repository is intact.
    goto :eof
)

goto :main

rem -----------------------------------------------------------------------------
rem Helper: prompt for virtualenv path selection
rem -----------------------------------------------------------------------------
:choose_venv_path
    set "FOUND_VENV=0"
    echo [info] Existing virtualenv directories in this repo:
    for /d %%D in ("%ROOT%\.venv*") do (
        set "FOUND_VENV=1"
        echo   - %%~fD
    )
    for /d %%D in ("%ROOT%\venv*") do (
        set "FOUND_VENV=1"
        echo   - %%~fD
    )
    if "!FOUND_VENV!"=="0" echo   (none detected)

    set "DEFAULT_VENV=%ROOT%\.venv"
    set /p "USER_VENV=Enter virtualenv path to use [%DEFAULT_VENV%]: "
    if /I "%USER_VENV%"=="quit" (
        set "VENV_DIR="
        exit /b 2
    )
    if "%USER_VENV%"=="" set "USER_VENV=%DEFAULT_VENV%"
    for %%I in ("%USER_VENV%") do set "VENV_DIR=%%~fI"
    exit /b 0

rem -----------------------------------------------------------------------------
rem Helper: run command and echo it
rem -----------------------------------------------------------------------------
:run_cmd
    set "CMD_LINE=%~1"
    if "%CMD_LINE%"=="" (
        echo [error] Internal script error: missing command.
        exit /b 1
    )
    echo → %CMD_LINE%
    call %CMD_LINE%
    exit /b %ERRORLEVEL%

rem -----------------------------------------------------------------------------
rem Helper: yes/no prompt
rem -----------------------------------------------------------------------------
:ask_yes_no
    set "PROMPT=%~1"
    set "DEFAULT=%~2"

    if /I "%DEFAULT%"=="Y" (
        set "SUFFIX= [Y/n]"
    ) else (
        set "SUFFIX= [y/N]"
    )

:ask_yes_no_loop
    set /p "ANSWER=%PROMPT%%SUFFIX% "
    if "%ANSWER%"=="" (
        if /I "%DEFAULT%"=="Y" (
            exit /b 0
        ) else (
            exit /b 1
        )
    )
    for %%I in (y Y yes YES Yes) do if "%%I"=="%ANSWER%" exit /b 0
    for %%I in (n N no NO No) do if "%%I"=="%ANSWER%" exit /b 1
    echo Please answer yes or no.
    goto :ask_yes_no_loop

rem -----------------------------------------------------------------------------
rem Helper: detect active env (conda or venv)
rem -----------------------------------------------------------------------------
:detect_active_env
    if defined CONDA_PREFIX (
        set "ACTIVE_ENV=conda"
        exit /b 0
    )
    if defined VIRTUAL_ENV (
        set "ACTIVE_ENV=venv"
        exit /b 0
    )
    set "ACTIVE_ENV="
    exit /b 1

rem -----------------------------------------------------------------------------
rem Ensure deps in current environment (pip install if needed)
rem -----------------------------------------------------------------------------
:ensure_current_env
    python -m pip show gradio >nul 2>&1
    if errorlevel 1 (
        echo [info] Gradio not found in the active environment.
        call :ask_yes_no "Install dependencies with pip install -r requirements.txt?" "Y"
        if errorlevel 1 (
            echo [error] Cannot continue without Gradio.
            exit /b 1
        )
        call :run_cmd "python -m pip install -r \"%REQUIREMENTS_TXT%\""
        if errorlevel 1 (
            echo [error] Installation failed.
            exit /b 1
        )
    )
    exit /b 0

rem -----------------------------------------------------------------------------
rem Setup virtualenv
rem -----------------------------------------------------------------------------
:setup_venv_env
    if exist "%VENV_DIR%\Scripts\python.exe" (
        echo [info] Virtual environment already exists at %VENV_DIR%
        call :ask_yes_no "Reinstall requirements inside the virtualenv?" "N"
        if not errorlevel 1 (
            call :run_cmd "\"%VENV_DIR%\Scripts\python.exe\" -m pip install -r \"%REQUIREMENTS_TXT%\""
            if errorlevel 1 exit /b 1
        )
        exit /b 0
    )

    echo [info] Creating virtual environment at %VENV_DIR%
    for %%I in ("%VENV_DIR%") do set "VENV_PARENT=%%~dpI"
    if not exist "%VENV_PARENT%" mkdir "%VENV_PARENT%" >nul 2>&1
    call :run_cmd "python -m venv \"%VENV_DIR%\""
    if errorlevel 1 exit /b 1

    echo [info] Installing dependencies inside the virtualenv
    call :run_cmd "\"%VENV_DIR%\Scripts\python.exe\" -m pip install --upgrade pip"
    if errorlevel 1 exit /b 1
    call :run_cmd "\"%VENV_DIR%\Scripts\python.exe\" -m pip install -r \"%REQUIREMENTS_TXT%\""
    if errorlevel 1 exit /b 1
    exit /b 0

rem -----------------------------------------------------------------------------
rem Main workflow
rem -----------------------------------------------------------------------------
:main
    call :detect_active_env
    if not errorlevel 1 (
        echo [info] Detected active Python environment (%ACTIVE_ENV%).
        call :ensure_current_env
        if errorlevel 1 goto :eof
        call :run_cmd "python \"%GRADIO_APP%\""
        goto :eof
    )

    echo [info] No active Python environment detected.
    call :choose_venv_path
    if errorlevel 2 (
        echo Aborting.
        goto :eof
    )
    if errorlevel 1 goto :eof
    call :setup_venv_env
    if errorlevel 1 goto :eof
    call :run_cmd "\"%VENV_DIR%\Scripts\python.exe\" \"%GRADIO_APP%\""
    goto :eof
