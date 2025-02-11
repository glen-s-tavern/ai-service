FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Создание и активация виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей в виртуальное окружение
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Открываем порт
EXPOSE 8000

# Запуск приложения через виртуальное окружение
CMD ["/opt/venv/bin/python", "main.py"] 