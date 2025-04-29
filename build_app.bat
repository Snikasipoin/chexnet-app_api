@echo off
echo ===============================
echo  Сборка проекта CheXNet в .exe
echo ===============================

:: 1. Активируем venv (если у тебя он называется "venv")
call venv\Scripts\activate

:: 2. Сборка
pyinstaller --noconfirm --onefile ^
--add-data "templates;templates" ^
--add-data "static;static" ^
--add-data "model.pth.tar;." ^
app.py

:: 3. Вывод
echo -------------------------------
echo Сборка завершена!
echo .exe файл находится в папке dist\
pause
