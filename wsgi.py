print("⚙️ Импорт Flask-приложения...")

try:
    from main import app
    print("✅ Flask-приложение успешно импортировано")
except Exception as e:
    print("❌ Ошибка при импорте Flask-приложения:")
    print(e)
    raise
