import json
import os
import sys
from elasticsearch import Elasticsearch
from models.fasttext_model import FastTextModel
from logger_config import setup_logger
from typing import List, Dict, Any, Optional

logger = setup_logger(__name__)

class TenderSearcher:
    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 es_scheme: str = "http",
                 es_api_key: Optional[str] = None):
        """
        Initialize the TenderSearcher with Elasticsearch connection and FastText model.

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

            # Verify connection with more detailed error handling
            try:
                if not self.es.ping():
                    raise ConnectionError("Elasticsearch cluster is not reachable!")
            except Exception as conn_error:
                logger.error(f"Connection error: {conn_error}")
                raise ConnectionError(f"Failed to connect to Elasticsearch at {es_host}:{es_port}") from conn_error

            self.model = FastTextModel()
            self.index_name = "tenders"

        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            raise

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search tenders by text query using full-text search.

        :param query: Text query to search
        :param top_k: Number of top results to return
        :return: List of matching tenders
        """
        body = {
            "query": {
                "match": {
                    "description": {
                        "query": query,
                        "analyzer": "russian_analyzer"
                    }
                }
            },
            "size": top_k
        }

        results = self.es.search(index=self.index_name, body=body)
        return [hit['_source'] for hit in results['hits']['hits']]

    def search_by_keyword(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        body = {
            "query": {
                "term": {
                    "description.keyword": query
                }
            },
            "size": top_k
        }
        results = self.es.search(index=self.index_name, body=body)
        return [hit['_source'] for hit in results['hits']['hits']]

    def search_by_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search tenders by vector similarity.

        :param query: Text query to convert to vector
        :param top_k: Number of top similar tenders to return
        :return: List of most similar tenders
        """

        # Convert query to vector
        query_vector = self.model.get_embeddings([query])[0]

        results = self.es.search(
                index=self.index_name,
                knn={
                    'field': 'vector',
                    'query_vector': query_vector.tolist(),
                    'k': top_k,
                    'num_candidates': top_k * 5,
                }
            )
        return [hit['_source'] for hit in results['hits']['hits']]

    def hybrid_search(self, query: str, top_k: int = 5, vector_weight: float = 0.5, text_weight: float = 0.5, prioritize_text: bool = True) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search combining text and vector similarity using Reciprocal Rank Fusion (RRF).

        :param query: Text query
        :param top_k: Number of top results to return
        :param vector_weight: Weight for vector similarity (0-1)
        :return: List of most relevant tenders
        """
        query = self.model.preprocess_text(query)
        # Perform text search
        text_results = self.search_by_text(query, top_k)
        text_ids = {result['tender_id']: result for result in text_results}

        # Perform vector search
        vector_results = self.search_by_vector(query, top_k)
        vector_ids = {result['tender_id']: result for result in vector_results}

        # Combine results using Reciprocal Rank Fusion (RRF)
        combined_scores = {}
        rrf_k = 60  # RRF constant to prevent division by zero

        # Score text search results
        for idx, result in enumerate(text_results, 1):
            tender_id = result['tender_id']
            text_score = 1 / (rrf_k + idx)
            combined_scores[tender_id] = combined_scores.get(tender_id, 0) + (text_score * text_weight)  # Reduced weight for text matches

        # Score vector search results with additional weight
        for idx, result in enumerate(vector_results, 1):
            tender_id = result['tender_id']
            vector_score = 1 / (rrf_k + idx)
            combined_scores[tender_id] = combined_scores.get(tender_id, 0) + (vector_score * vector_weight)

        # Sort results by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        # Retrieve top_k results, prioritizing results based on configuration
        final_results = []
        search_type = []
        scores = []

        if prioritize_text:
            # Prioritize text search results first
            for tender_id, _ in sorted_results:
                if tender_id in text_ids:
                    final_results.append(text_ids[tender_id])
                    search_type.append("text")
                    scores.append(combined_scores[tender_id])
                if len(final_results) == top_k:
                    break

            # If not enough results from text search, supplement with vector search results
            if len(final_results) < top_k:
                for tender_id, _ in sorted_results:
                    if tender_id in vector_ids and tender_id not in text_ids:
                        final_results.append(vector_ids[tender_id])
                        search_type.append("vector")
                        scores.append(combined_scores[tender_id])
                    if len(final_results) == top_k:
                        break
        else:
            # Use combined results without strict prioritization
            for tender_id, _ in sorted_results:
                if tender_id in text_ids or tender_id in vector_ids:
                    if tender_id in text_ids:
                        final_results.append(text_ids[tender_id])
                        search_type.append("text")
                    else:
                        final_results.append(vector_ids[tender_id])
                        search_type.append("vector")
                    scores.append(combined_scores[tender_id])
                if len(final_results) == top_k:
                    break

        return final_results, search_type, scores

def main():
    # Docker-friendly environment variables
    es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    es_port = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    es_scheme = os.getenv('ELASTICSEARCH_SCHEME', 'http')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')

    try:
        # Initialize searcher with configurable parameters
        searcher = TenderSearcher(
            es_host=es_host,
            es_port=es_port,
            es_scheme=es_scheme,
            es_api_key=es_api_key
        )

        # Example search queries
        search_queries = [
            "поставка огурцов",
            "огурцы",
            "дизайн",
            "тапки",
            "поставка автобусов",
            "поставка товаров для вертолета",
            "самолет",
            "отправка смс",
            "создание вебсайтов",
            "вебсайты",
            "SMS",
        ]

        print("Demonstrating different search methods:\n")

        for query in search_queries:
            print(f"Search Query: '{query}'\n")
            '''
            print("1. Text Search Results:")
            text_results = searcher.search_by_text(query, 30)
            for result in text_results:
                print(f"- {result['tender_id']}: {result['description']}")
            print("\n")

            print("2. Vector Similarity Search Results:")
            vector_results = searcher.search_by_vector(query, 30)
            for result in vector_results:
                print(f"- {result['tender_id']}: {result['description']}")
            print("\n")
            '''
            print("3. Hybrid Search Results:")
            hybrid_results = searcher.hybrid_search(query, 30, vector_weight=0.3, text_weight=0.5, prioritize_text=False)
            for result, search_type, score in zip(hybrid_results[0], hybrid_results[1], hybrid_results[2]):
                print(f"- {score} - {search_type} - {result['description']} - {result['tender_id']}")
            print("\n" + "="*50 + "\n")

    except ConnectionError as e:
        print(f"Connection Error: {e}")
        print("Please ensure Elasticsearch is running:")
        print("1. Start with Docker: docker-compose up -d elasticsearch")
        print("2. Check container status: docker-compose ps")
        print("3. View logs: docker-compose logs elasticsearch")
        sys.exit(1)

if __name__ == "__main__":
    main()