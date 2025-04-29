# wsgi.py
print("⚙️ Импорт Flask-приложения...")

from app import app  # Импортирует объект Flask

print("✅ Flask-приложение успешно импортировано")

if __name__ == "__main__":
    app.run()