import os
import json
from typing import List, Optional
from datetime import datetime
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from annoy import AnnoyIndex
from download_tenders import (
    download_tenders,
    OUTPUT_DIR,
)
from logger_config import setup_logger
from models import ModelFactory, ModelType, BaseEmbeddingModel
import numpy as np

logger = setup_logger(__name__)

# Ensure resources directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Tenders Search API",
    description="API for downloading and searching tenders from zakupki.gov.ru",
    version="1.0.0"
)

class TenderDownloadRequest(BaseModel):
    regions: Optional[List[str]] = Field(
        default=["77", "78"],
        description="List of region codes to download tenders from"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for tender search (YYYY-MM-DD)",
        example="2024-10-01"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for tender search (YYYY-MM-DD)",
        example="2024-10-02"
    )
    vectorize: Optional[bool] = Field(
        default=True,
        description="Create vector embeddings for downloaded tenders"
    )
    model_type: Optional[ModelType] = Field(
        default=ModelType.roberta,
        description="Model to use for vectorization: 'roberta' or 'fasttext'"
    )

class TenderSearchRequest(BaseModel):
    query: str = Field(
        description="Text query to search for similar tenders"
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of similar tenders to return"
    )
    model_type: ModelType = Field(
        default=ModelType.roberta,
        description="Model to use for search: 'roberta' or 'fasttext'"
    )

class TenderSearchResult(BaseModel):
    purchase_number: str
    tender_name: str
    similarity_score: float

def angular_distance_to_similarity(distance: float) -> float:
    """
    Convert angular distance to similarity score.
    For angular distance, similarity = cos(θ) = 1 - distance
    Normalizes to range [0, 1]
    """
    similarity = 1 - (distance / 2)  # Convert angular distance to cosine similarity
    return max(0.0, min(1.0, similarity))  # Ensure score is between 0 and 1

def create_tender_embeddings(tenders_summary_path: str, model: BaseEmbeddingModel):
    """Create embeddings for tenders using the specified model."""
    try:
        with open(tenders_summary_path, 'r', encoding='utf-8') as f:
            tenders = json.load(f)

        tender_texts = list(tenders.values())
        tender_keys = list(tenders.keys())

        logger.info(f"Building embeddings using {model.__class__.__name__}...")
        embeddings = model.get_embeddings(tender_texts)

        if embeddings is None:
            raise ValueError("Failed to generate embeddings")

        # Create and save index
        annoy_index = AnnoyIndex(model.embedding_dim, 'angular')
        tenders_metadata = {}

        for i, (embedding, key, text) in enumerate(zip(embeddings, tender_keys, tender_texts)):
            annoy_index.add_item(i, embedding)
            tenders_metadata[i] = {
                'purchase_number': key,
                'tender_name': text
            }

        annoy_index.build(10)
        model_name = model.__class__.__name__.lower().replace('model', '')
        annoy_index.save(f'{OUTPUT_DIR}/tenders_index_{model_name}.ann')

        with open(f'{OUTPUT_DIR}/tenders_metadata_{model_name}.json', 'w', encoding='utf-8') as f:
            json.dump(tenders_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Created ANNOY index with {len(tenders_metadata)} tenders")
        return True
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return False

@app.post("/download-tenders/")
async def api_download_tenders(request: TenderDownloadRequest):
    try:
        if request.start_date:
            datetime.strptime(request.start_date, '%Y-%m-%d')
        if request.end_date:
            datetime.strptime(request.end_date, '%Y-%m-%d')

        # Download tenders
        download_tenders(
            regions=request.regions,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Create embeddings if requested
        if request.vectorize:
            model = ModelFactory.get_model(request.model_type)
            success = create_tender_embeddings(
                os.path.join(OUTPUT_DIR, 'tenders_summary.json'),
                model
            )
            if not success:
                raise Exception("Failed to create embeddings")

        # Return results
        summary_path = os.path.join(OUTPUT_DIR, 'tenders_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                tenders_summary = json.load(f)
            return {
                "message": "Tenders downloaded successfully",
                "total_tenders": len(tenders_summary),
                "summary": tenders_summary,
                "model_type": request.model_type
            }
        else:
            return {"message": "Tenders downloaded, but no summary found"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading tenders: {str(e)}")

@app.post("/search-tenders/",
          response_model=List[TenderSearchResult])
async def api_search_tenders(request: TenderSearchRequest):
    try:
        logger.info(f"Searching tenders with query: '{request.query}', top_k: {request.top_k}, model: {request.model_type}")

        # Get model instance
        model = ModelFactory.get_model(request.model_type)
        model_name = model.__class__.__name__.lower().replace('model', '')

        # Check if index exists
        index_path = os.path.join(OUTPUT_DIR, f'tenders_index_{model_name}.ann')
        metadata_path = os.path.join(OUTPUT_DIR, f'tenders_metadata_{model_name}.json')

        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            raise HTTPException(
                status_code=400,
                detail=f"Embeddings not found for model {request.model_type}. Run download with --vectorize and specified model type first."
            )

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            tenders_metadata = json.load(f)

        # Generate query embedding
        query_embedding = model.get_embeddings([request.query])
        if query_embedding is None or query_embedding.size == 0:
            raise ValueError("Failed to generate embedding for query")

        # Load and search index
        annoy_index = AnnoyIndex(model.embedding_dim, 'angular')
        annoy_index.load(index_path)

        similar_indices = annoy_index.get_nns_by_vector(
            query_embedding[0], request.top_k, include_distances=True
        )

        # Format results
        results = []
        for index, distance in zip(similar_indices[0], similar_indices[1]):
            metadata = tenders_metadata.get(str(index), {})
            if metadata:
                results.append(
                    TenderSearchResult(
                        purchase_number=metadata.get('purchase_number', 'N/A'),
                        tender_name=metadata.get('tender_name', 'N/A'),
                        similarity_score=angular_distance_to_similarity(distance)
                    )
                )

        logger.info(f"Returning {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching tenders: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "tenders_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )