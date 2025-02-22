import sqlite3
from contextlib import contextmanager
import os
from typing import List, Dict, Any

import tqdm
from src.models.model_factory import ModelFactory
from src.logger_config import setup_logger
import numpy as np

logger = setup_logger(__name__)

class Database:
    def __init__(self, db_path: str = "resources/tenders.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.create_tables()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def create_tables(self):
        """Создание таблиц в базе данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Создаем таблицу для тендеров
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tenders (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    price DECIMAL(20, 2),
                    law_type TEXT,           
                    purchase_method TEXT,
                    okpd2_code TEXT,          
                    
                    publish_date TEXT,        
                    end_date TEXT,            
                    
                    customer_inn TEXT,        
                    customer_name TEXT,       
                    
                    region TEXT NOT NULL,
                    date_added TEXT NOT NULL,
                    vector BLOB
                )
            """)
            
            # Создаем индексы для ускорения поиска
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_region_date ON tenders(region, date_added)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON tenders(price)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_law_type ON tenders(law_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_purchase_method ON tenders(purchase_method)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_inn ON tenders(customer_inn)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_okpd2 ON tenders(okpd2_code)")
            
            conn.commit()

    def insert_tenders(self, tenders: List[Dict[str, Any]], region: str, date: str) -> int:
        """
        Вставка тендеров в базу данных
        
        :param tenders: Список тендеров
        :param region: Код региона
        :param date: Дата в формате YYYY-MM-DD
        :return: Количество добавленных тендеров
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            count = 0
            
            for tender in tenders:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO tenders (
                            id, name, price, law_type, purchase_method,
                            okpd2_code, publish_date, end_date,
                            customer_inn, customer_name,
                            region, date_added
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        tender['id'],
                        tender['name'],
                        tender.get('price'),
                        tender.get('law_type'),
                        tender.get('purchase_method'),
                        tender.get('okpd2_code'),
                        tender.get('publish_date'),
                        tender.get('end_date'),
                        tender.get('customer_inn'),
                        tender.get('customer_name'),
                        region,
                        date
                    ))
                    count += 1
                except sqlite3.Error as e:
                    logger.error(f"Error inserting tender {tender['id']}: {e}")
                    continue
            
            conn.commit()
            return count

    def get_tenders(
        self, 
        region: str = None, 
        date: str = None, 
        min_price: float = None, 
        max_price: float = None,
        law_type: str = None,
        purchase_method: str = None,
        okpd2_code: str = None,
        customer_inn: str = None,
        customer_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Получение тендеров с опциональной фильтрацией по региону, дате публикации и другим параметрам

        :param region: Регион тендера (опционально)
        :param date: Дата для сравнения с датой публикации тендера (опционально)
        :param min_price: Минимальная цена тендера (опционально)
        :param max_price: Максимальная цена тендера (опционально)
        :param law_type: Тип закона (опционально)
        :param purchase_method: Метод закупки (опционально)
        :param okpd2_code: Код ОКПД2 (опционально)
        :param customer_inn: ИНН заказчика (опционально)
        :return: Список тендеров, соответствующих условиям
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare the base query
            query = """
                SELECT 
                    id, name, price, law_type, purchase_method, 
                    okpd2_code, publish_date, end_date, 
                    customer_inn, customer_name, 
                    region, date_added, vector
                FROM tenders 
                WHERE 1=1
            """
            
            # Prepare parameters
            params = []
            if customer_name is not None:
                query += " AND UPPER(customer_name) LIKE ?"
                params.append(f"%{customer_name.upper()}%")

            # Add region filter if provided
            if region is not None:
                query += " AND region = ?"
                params.append(region)
            
            # Add publish_date filter if provided
            if date is not None:
                query += " AND publish_date > ?"
                params.append(date)
            
            # Add price filters if provided
            if min_price is not None:
                query += " AND price >= ?"
                params.append(min_price)
            
            if max_price is not None:
                query += " AND price <= ?"
                params.append(max_price)
            
            # Add law_type filter if provided
            if law_type is not None:
                # Split law_type into words and create OR conditions for each word
                words = law_type.split()
                query += " AND ("
                query += " OR ".join(["UPPER(law_type) LIKE ?" for _ in words])
                query += ")"
                params.extend([f"%{word.upper()}%" for word in words])
            
            # Add purchase_method filter if provided
            if purchase_method is not None:
                # Split purchase_method into words and create OR conditions for each word
                words = purchase_method.split()
                query += " AND ("
                query += " OR ".join(["UPPER(purchase_method) LIKE ?" for _ in words])
                query += ")"
                params.extend([f"%{word.upper()}%" for word in words])
            
            # Add okpd2_code filter if provided
            if okpd2_code is not None:
                query += " AND okpd2_code = ?"
                params.append(okpd2_code)
            
            # Add customer_inn filter if provided
            if customer_inn is not None:
                query += " AND customer_inn = ?"
                params.extend(customer_inn)
            
            # Execute the query
            cursor.execute(query, params)
            
            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'price': row[2],
                    'law_type': row[3],
                    'purchase_method': row[4],
                    'okpd2_code': row[5],
                    'publish_date': row[6],
                    'end_date': row[7],
                    'customer_inn': row[8],
                    'customer_name': row[9],
                    'region': row[10],
                    'date_added': row[11],
                    'vector': np.frombuffer(row[12], dtype=np.float32) if row[12] is not None else None
                }
                for row in cursor.fetchall()
            ]

    def get_tenders_with_null_vector(self):
        with  self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, price FROM tenders WHERE vector IS NULL")
            return [
                {
                    'id': row[0],
                    'name': row[1],
                }
                for row in cursor.fetchall()
            ]
    
    def update_tender_vectors(self, ids: List[str], vectors: List[List[float]]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Convert vectors to binary format for SQLite BLOB storage
            for tender_id, vector in zip(ids, vectors):
                # Convert vector to bytes using numpy's serialization
                vector_blob = sqlite3.Binary(np.array(vector).tobytes())
                cursor.execute("UPDATE tenders SET vector = ? WHERE id = ?", (vector_blob, tender_id))
            conn.commit()

    def vectorize_tenders(self):
        model = ModelFactory.get_model('roberta')
        tenders = self.get_tenders_with_null_vector()
        batch_size = 100
        for i in tqdm.tqdm(range(0, len(tenders), batch_size), desc="Vectorizing tenders", unit="batch"):
            batch = tenders[i:i + batch_size]
            ids = [tender['id'] for tender in batch]
            names = [tender['name'] for tender in batch]
            vectors = model.get_embeddings(names)
            self.update_tender_vectors(ids, vectors)

    def add_vector_column(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("ALTER TABLE tenders ADD COLUMN vector BLOB")
            conn.commit()
            
    def get_tender_vector(self, tender_id: str) -> np.ndarray:
        """Retrieve the vector for a specific tender"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector FROM tenders WHERE id = ?", (tender_id,))
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                # Convert bytes back to numpy array
                return np.frombuffer(result[0], dtype=np.float32)
            return None
            