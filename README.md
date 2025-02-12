1. Установить torch под свою систему: https://pytorch.org/get-started/locally/
2. Установить остальные библиотеки: `pip install -r requirements.txt`
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



