import json
import os
import sys
from elasticsearch import Elasticsearch
from models.fasttext_model import FastTextModel
from logger_config import setup_logger
from typing import List, Dict, Any, Optional
from tqdm import tqdm


logger = setup_logger(__name__)

class TenderIndexer:
    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 es_scheme: str = "http",
                 es_api_key: Optional[str] = None):
        """
        Initialize the TenderIndexer with Elasticsearch connection and FastText model.

        :param es_host: Elasticsearch host
        :param es_port: Elasticsearch port
        :param es_scheme: Connection scheme (http/https)
        :param es_api_key: Optional API key for authentication
        """
        try:
            # Configure Elasticsearch connection
            connection_params = {
                'hosts': [{'host': es_host, 'port': es_port, 'scheme': es_scheme}],
                'retry_on_timeout': True,
                'max_retries': 3,
                'timeout': 10
            }

            # Add API key if provided
            if es_api_key:
                connection_params['api_key'] = es_api_key

            # Create Elasticsearch client
            self.es = Elasticsearch(**connection_params)

            # Verify connection
            if not self.es.ping():
                raise ConnectionError("Elasticsearch cluster is not reachable!")

            self.model = FastTextModel()
            self.index_name = "tenders"

        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            raise

    def create_index(self):
        """
        Create Elasticsearch index with proper mapping for tender data and vector fields.
        source: https://habr.com/ru/companies/otus/articles/844978/
        """
        mapping = {
            "mappings": {
                "properties": {
                    "tender_id": {
                        "type": "keyword"  # Exact match for tender ID
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "russian_analyzer",  # Use custom Russian analyzer for full-text search
                        "fields": {
                            "raw": {
                                "type": "text",  # Additional text field for different analysis
                                "analyzer": "russian_analyzer"
                            }
                        }
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.model.embedding_dim,
                        "index": True,
                        "similarity": "cosine"  # Cosine similarity for vector search
                    }
                }
            },
            "settings": {
                "index": {
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "b": 0.3,  # Reduced b value to minimize impact of term frequency
                            "k1": 0.8   # Lower k1 to reduce importance of common terms
                        }
                    }
                },
                "analysis": {
                    "analyzer": {
                        "russian_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "russian_stop",  # Built-in Russian stopwords
                                "russian_stemmer"  # Russian language stemmer
                            ]
                        }
                    },
                    "filter": {
                        "russian_stop": {
                            "type": "stop",
                            "stopwords": "Поставка"
                        },
                        "russian_stemmer": {
                            "type": "stemmer",
                            "language": "russian"
                        }
                    }
                }
            }
        }

        '''"index": {
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "b": 0.9,  # Reduced b value for less term frequency impact
                            "k1": 1.2   # Adjusted k1 for fine-tuned relevance
                        }
                    }
                },'''

        # Delete index if it exists
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        # Create new index
        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created index '{self.index_name}' with vector mapping")

    def load_tenders(self, file_path: Optional[str] = None) -> Dict[str, str]:
        """
        Load tenders from JSON file.

        :param file_path: Optional path to tenders JSON file
        :return: Dictionary of tenders
        """
        if file_path is None:
            file_path = os.path.join("resources", "tenders_data", "tenders_summary.json")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Tenders file not found: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in tenders file: {file_path}")
            raise

    def index_tenders(self, tenders: Optional[Dict[str, str]] = None, batch_size: int = 1000):
        """
        Index tenders with their vector representations.

        :param tenders: Optional dictionary of tenders to index
        :param batch_size: Number of tenders to index in each batch
        """
        # Load tenders if not provided
        if tenders is None:
            tenders = self.load_tenders()

        logger.info(f"Loaded {len(tenders)} tenders")

        # Create index with proper mapping
        self.create_index()

        # Convert tenders to list of items for easier batching
        tender_items = list(tenders.items())

        # Process tenders in batches
        for i in tqdm(range(0, len(tender_items), batch_size), desc="Indexing Tenders"):
            batch = tender_items[i:i + batch_size]

            # Get descriptions for the batch
            descriptions = [self.model.preprocess_text(desc) for _, desc in batch]
            descriptions = [desc for _, desc in batch]

            # Generate vectors for the batch
            vectors = self.model.get_embeddings(descriptions)

            # Prepare bulk indexing actions
            actions = []
            for ((tender_id, description), vector) in zip(batch, vectors):
                description = self.model.preprocess_text(description)
                doc = {
                    'tender_id': tender_id,
                    'description': description,
                    'vector': vector.tolist()
                }
                actions.append({'index': {'_index': self.index_name, '_id': tender_id}})
                actions.append(doc)

            # Bulk index the batch
            if actions:
                self.es.bulk(index=self.index_name, body=actions, refresh=True)

        logger.info("Finished indexing all tenders")

def main():
    # Environment variables for configuration
    es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    es_port = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    es_scheme = os.getenv('ELASTICSEARCH_SCHEME', 'http')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')

    try:
        # Initialize indexer
        indexer = TenderIndexer(
            es_host=es_host,
            es_port=es_port,
            es_scheme=es_scheme,
            es_api_key=es_api_key
        )

        # Index tenders
        indexer.index_tenders()

        logger.info("Successfully indexed all tenders")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()