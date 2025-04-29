print("⚙️ Импорт Flask-приложения...")

from main import app  # <-- меняем app → main

print("✅ Flask-приложение успешно импортировано")

if __name__ == "__main__":
    app.run()