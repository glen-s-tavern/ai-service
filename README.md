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
          "start_date": "2024-01-01", 
          "end_date": "2024-01-31", 
          "vectorize": true
        }'
```

2. Найти похожие тендеры по текстовому запросу в созданной базе:
```bash
curl -X POST "http://localhost:8000/search-tenders/" \
     -H "Content-Type: application/json" \
     -d '{
            "query": "Медицинские услуги", 
            "top_k": 5
         }'
```



