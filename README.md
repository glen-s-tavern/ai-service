1. Установить torch под свою систему: https://pytorch.org/get-started/locally/
2. Установить остальные библиотеки: `pip install -r requirements.txt`
3. Скачать модель fasttext: `python -c "from huggingface_hub import hf_hub_download;hf_hub_download(repo_id="facebook/fasttext-ru-vectors", filename="model.bin", local_dir='resources')"`
3. Запустить: `uvicorn tenders_api:app --reload`

Дальше есть 2 функции:

1. Скачать тендеры и создать базу по списку регионов и промежутку дат:
```bash
curl -X POST "http://localhost:8000/download-tenders/" \
     -H "Content-Type: application/json" \
     -d '{
          "regions": ["77", "78"],
          "start_date": "2024-10-01",
          "end_date": "2024-10-02",
          "vectorize": true,
          "model_type": "roberta"  # Можно выбрать "roberta" или "fasttext"
        }'
```

2. Найти похожие тендеры по текстовому запросу в созданной базе:
```bash
curl -X POST "http://localhost:8000/search-tenders/" \
     -H "Content-Type: application/json" \
     -d '{
            "query": "Медицинские услуги",
            "top_k": 5,
            "model_type": "roberta"  # Используйте ту же модель, что и при создании базы
         }'
```

Доступные модели для векторизации:
- `roberta`: RuRoBERTa - трансформер модель, обученная на русском языке
- `fasttext`: FastText - модель на основе подсловных n-грамм, обученная на Википедии и новостях. Лучше справляется с редкими и неизвестными словами благодаря использованию подсловной информации.

# Tenders Search Service

## Elasticsearch Setup with Docker

### Prerequisites
- Docker
- Docker Compose

### Running Elasticsearch

1. Start Elasticsearch:
```bash
docker-compose up -d elasticsearch
```

2. Verify Elasticsearch is running:
```bash
docker-compose ps
# or
curl http://localhost:9200
```

### Indexing Tenders

1. Run the indexing script:
```bash
python index_tenders.py
```

### Searching Tenders

1. Run the search script:
```bash
python search_tenders.py
```

### Stopping Elasticsearch

```bash
docker-compose down
```

### Troubleshooting

- Ensure Docker is running
- Check container logs:
```bash
docker-compose logs elasticsearch
```

### Environment Variables

You can configure Elasticsearch connection using:
- `ELASTICSEARCH_HOST`
- `ELASTICSEARCH_PORT`
- `ELASTICSEARCH_SCHEME`

Example:
```bash
export ELASTICSEARCH_HOST=localhost
export ELASTICSEARCH_PORT=9200
```



