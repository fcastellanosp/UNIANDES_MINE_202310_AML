@echo off
cd..
cd..
setlocal
set PROJECTPATH=%cd%
set PYTHONPATH=%PYTHON3_HOME%;%cd%
echo "Starting the app at '%PROJECTPATH%\src\front\main.py"
echo "Enviroment variable '%PYTHONPATH%"
%PYTHON3_HOME%\python -m streamlit run %PROJECTPATH%\main.py
endlocal