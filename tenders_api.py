import os
import json
from typing import List, Optional
from datetime import datetime
import torch
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from annoy import AnnoyIndex
from download_tenders import (
    download_tenders, 
    create_tender_embeddings, 
    generate_embeddings_batch,
    OUTPUT_DIR,
    EMBEDDING_DIM
)
import logging

# Setup logging with a more specific configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', # Force override any existing logger configuration
)

# Create a logger specific to our application
logger = logging.getLogger(__name__)

# Ensure handlers are cleared to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Global variables for model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None

def init_model():
    global tokenizer, model
    try:
        logger.info("Initializing RuRoBERTa model...")
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruRoberta-large")
        model = AutoModel.from_pretrained("ai-forever/ruRoberta-large")
        model = model.to(device)
        model.eval()
        logger.info(f"Model initialized successfully on {device}")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

app = FastAPI(
    title="Tenders Search API",
    description="API for downloading and searching tenders from zakupki.gov.ru",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    init_model()

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


class TenderSearchRequest(BaseModel):
    query: str = Field(
        description="Text query to search for similar tenders"
    )
    top_k: Optional[int] = Field(
        default=5, 
        description="Number of similar tenders to return"
    )


class TenderSearchResult(BaseModel):
    purchase_number: str
    tender_name: str
    similarity_score: float

@app.post("/download-tenders/", 
          summary="Download tenders from zakupki.gov.ru",
          description="Download tenders for specified regions and date range")


async def api_download_tenders(request: TenderDownloadRequest):
    try:
        if request.start_date:
            datetime.strptime(request.start_date, '%Y-%m-%d')
        if request.end_date:
            datetime.strptime(request.end_date, '%Y-%m-%d')
        download_tenders(
            regions=request.regions, 
            start_date=request.start_date, 
            end_date=request.end_date
        )
        if request.vectorize:
            create_tender_embeddings(os.path.join(OUTPUT_DIR, 'tenders_summary.json'))
        summary_path = os.path.join(OUTPUT_DIR, 'tenders_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                tenders_summary = json.load(f)
            return {
                "message": "Tenders downloaded successfully", 
                "total_tenders": len(tenders_summary),
                "summary": tenders_summary
            }
        else:
            return {"message": "Tenders downloaded, but no summary found"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading tenders: {str(e)}")
    

@app.post("/search-tenders/", 
          response_model=List[TenderSearchResult],
          summary="Search similar tenders",
          description="Find similar tenders using semantic search")
async def api_search_tenders(request: TenderSearchRequest):
    try:
        logger.info(f"Searching tenders with query: '{request.query}', top_k: {request.top_k}")
        
        index_path = os.path.join(OUTPUT_DIR, 'tenders_index.ann')
        metadata_path = os.path.join(OUTPUT_DIR, 'tenders_metadata.json')
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            raise HTTPException(
                status_code=400, 
                detail="Embeddings not found. Run download with --vectorize first."
            )
            
        logger.info(f"Loading index from {index_path}")
        logger.info(f"Loading metadata from {metadata_path}")
        
        # Load metadata and index
        with open(metadata_path, 'r', encoding='utf-8') as f:
            tenders_metadata = json.load(f)
        
        annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')
        annoy_index.load(index_path)
        
        # Generate query embedding
        query_embedding = generate_embeddings_batch([request.query], tokenizer, model, device)
        
        if query_embedding is None or query_embedding.size == 0:
            logger.error("Failed to generate embedding for query")
            return []
        
        # Search similar tenders
        similar_indices = annoy_index.get_nns_by_vector(
            query_embedding[0], request.top_k, include_distances=True
        )
        
        results = []
        for index, distance in zip(similar_indices[0], similar_indices[1]):
            metadata = tenders_metadata.get(str(index), {})
            if metadata:
                results.append(
                    TenderSearchResult(
                        purchase_number=metadata.get('purchase_number', 'N/A'),
                        tender_name=metadata.get('tender_name', 'N/A'),
                        similarity_score=float(1 - distance)  # Convert distance to similarity score
                    )
                )
        
        logger.info(f"Returning {len(results)} results")
        for result in results:
            logger.info(f"Score: {result.similarity_score:.4f} - {result.purchase_number}")
            
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