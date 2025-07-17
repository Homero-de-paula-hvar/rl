@echo off
REM Script de setup para Windows - AutoRL Framework

REM Verifica se Python 3.10 está instalado
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo Python launcher (py) não encontrado. Instale Python 3.10 ou 3.11 e tente novamente.
    pause
    exit /b 1
)

REM Cria ambiente virtual com Python 3.10
py -3.10 -m venv venv
if %errorlevel% neq 0 (
    echo Erro ao criar ambiente virtual com Python 3.10. Verifique se o Python 3.10 está instalado.
    pause
    exit /b 1
)

REM Ativa o ambiente virtual
call venv\Scripts\activate.bat

REM Instala numpy, pandas, scikit-learn primeiro (wheels)
pip install --upgrade pip
pip install numpy==1.24.4 pandas==2.0.3 scikit-learn==1.3.0

REM Instala o restante das dependências
pip install -r framework-main\requirements.txt

REM Mensagem final
echo Ambiente pronto! Para rodar a aplicação Flask, use:
echo.
echo     python framework-main\web_app.py
echo.
echo Acesse http://localhost:8080 no navegador.
pause 