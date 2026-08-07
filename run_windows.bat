@echo off
REM ===================================================================
REM  NeoRefacer - execucao local no Windows com DirectML (GPU AMD/Intel)
REM
REM  Primeiro uso: cria a venv e instala tudo (demora alguns minutos).
REM  Usos seguintes: sobe o app direto.
REM
REM  Nao afeta o Colab: la o provider CUDA continua sendo escolhido.
REM ===================================================================

REM  EnableDelayedExpansion e obrigatorio: sem ele o "if not defined" dentro do
REM  laco abaixo seria avaliado uma unica vez, na analise do bloco, e a ultima
REM  versao testada venceria em vez da primeira encontrada.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set VENV_DIR=.venv-dml
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

REM --- Localiza um Python compativel ---
REM  insightface 0.7.3 (a versao usada no Colab) so tem wheel ate o Python
REM  3.12. No 3.13 o pip tenta compilar do fonte e falha pedindo o Visual
REM  Studio Build Tools, por isso a versao e fixada aqui em vez de usar o
REM  "python" do PATH, que costuma apontar para a instalacao mais recente.
set PY_LAUNCHER=
for %%V in (3.11 3.12 3.10) do (
    if not defined PY_LAUNCHER (
        py -%%V -c "import sys" >nul 2>nul && set "PY_LAUNCHER=py -%%V"
    )
)

REM  Fallback: sem o launcher "py" instalado, tenta o python do PATH e aceita
REM  apenas se a versao for 3.10-3.12.
if not defined PY_LAUNCHER (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>nul && set "PY_LAUNCHER=python"
)

if not defined PY_LAUNCHER (
    echo.
    echo [ERRO] Nenhum Python compativel ^(3.10, 3.11 ou 3.12^) foi encontrado.
    echo.
    echo Baixe o Python 3.11 em:
    echo   https://www.python.org/downloads/release/python-3119/
    echo Escolha "Windows installer ^(64-bit^)".
    echo IMPORTANTE: marque "Add python.exe to PATH" durante a instalacao.
    echo.
    echo Obs.: o Python 3.13 nao serve para este projeto.
    pause
    exit /b 1
)

echo Usando: %PY_LAUNCHER%

REM --- venv completa? o marcador so e escrito ao final de uma instalacao bem
REM     sucedida, entao uma venv interrompida no meio e refeita do zero em vez
REM     de seguir para o app com dependencias faltando. ---
set INSTALL_MARKER=%VENV_DIR%\.install-complete

if exist "%VENV_DIR%" if not exist "%INSTALL_MARKER%" (
    echo.
    echo === Instalacao anterior incompleta: recriando o ambiente ===
    rmdir /s /q "%VENV_DIR%"
)

if not exist "%INSTALL_MARKER%" (
    echo.
    echo === Primeira execucao: criando ambiente virtual ===
    %PY_LAUNCHER% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERRO] Falha ao criar a venv.
        pause
        exit /b 1
    )

    echo.
    echo === Instalando dependencias ^(pode demorar alguns minutos^) ===
    "%PYTHON_EXE%" -m pip install --upgrade pip

    REM  Wheel pre-compilada do insightface — ver scripts/install_insightface_win.py
    REM  para o motivo. A montagem da URL fica no script Python porque depende da
    REM  tag de ABI do interpretador, e resolver isso no batch exigiria um for /f
    REM  sobre um comando entre aspas, cujo escape e fonte recorrente de erro.
    echo.
    echo === Instalando insightface ^(wheel pre-compilada^) ===
    "%PYTHON_EXE%" scripts\install_insightface_win.py
    if errorlevel 1 (
        echo.
        echo [ERRO] Falha ao instalar o insightface.
        echo Verifique a conexao com a internet e rode o script de novo.
        pause
        exit /b 1
    )

    "%PYTHON_EXE%" -m pip install -r requirements-DML.txt
    if errorlevel 1 (
        echo.
        echo [ERRO] Falha ao instalar as dependencias.
        echo Apague a pasta %VENV_DIR% e rode este script de novo.
        pause
        exit /b 1
    )

    echo ok> "%INSTALL_MARKER%"
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
