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

# Установка PyTorch и зависимостей в виртуальное окружение
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir torch  --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Создаем директорию для монтирования модели
RUN mkdir -p /app/resources

# Копирование исходного кода (без resources)
COPY . .

# Открываем порт
EXPOSE 8000

# Запуск приложения через uvicorn
CMD ["/opt/venv/bin/uvicorn", "tenders_api:app", "--host", "0.0.0.0", "--port", "8000"]