@echo off
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/chexnet_app_project.git
git push -u origin main
echo ====== ЗАЛИТО НА GITHUB ======
echo Теперь задеплой на Timeweb Apps. Не забудь указать Python 3.9.13 и переменные окружения.
pause
