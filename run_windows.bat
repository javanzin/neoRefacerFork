@echo off
REM ===================================================================
REM  NeoRefacer - execucao local no Windows com DirectML (GPU AMD/Intel)
REM
REM  Primeiro uso: cria a venv e instala tudo (demora alguns minutos).
REM  Usos seguintes: sobe o app direto.
REM
REM  Nao afeta o Colab: la o provider CUDA continua sendo escolhido.
REM ===================================================================

setlocal
cd /d "%~dp0"

set VENV_DIR=.venv-dml
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

REM --- Python instalado? ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    echo Instale o Python 3.10 ou 3.11 em https://www.python.org/downloads/
    echo IMPORTANTE: marque "Add Python to PATH" durante a instalacao.
    pause
    exit /b 1
)

REM --- venv existe? se nao, cria e instala ---
if not exist "%PYTHON_EXE%" (
    echo.
    echo === Primeira execucao: criando ambiente virtual ===
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERRO] Falha ao criar a venv.
        pause
        exit /b 1
    )

    echo.
    echo === Instalando dependencias ^(pode demorar alguns minutos^) ===
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r requirements-DML.txt
    if errorlevel 1 (
        echo.
        echo [ERRO] Falha ao instalar as dependencias.
        echo Apague a pasta %VENV_DIR% e rode este script de novo.
        pause
        exit /b 1
    )
    echo.
    echo === Instalacao concluida ===
)

REM --- REFACER_PROFILE=1 imprime o tempo por estagio no fim do processamento
REM     ^(deteccao / embedding / swap^), para comparar com o Colab. ---
set REFACER_PROFILE=1

echo.
echo === Iniciando NeoRefacer ===
echo Abra no navegador: http://127.0.0.1:7860
echo Feche esta janela para encerrar.
echo.

"%PYTHON_EXE%" app.py %*

if errorlevel 1 (
    echo.
    echo [ERRO] O aplicativo terminou com erro. Veja a mensagem acima.
    pause
    exit /b 1
)

endlocal
