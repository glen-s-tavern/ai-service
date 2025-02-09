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
    search_similar_tenders,
    load_ruroberta_model,
    generate_embeddings_batch,
    OUTPUT_DIR,
    EMBEDDING_DIM
)

app = FastAPI(
    title="Tenders Search API",
    description="API for downloading and searching tenders from zakupki.gov.ru",
    version="1.0.0"
)


class TenderDownloadRequest(BaseModel):
    regions: Optional[List[str]] = Field(
        default=["77"], 
        description="List of region codes to download tenders from"
    )
    start_date: Optional[str] = Field(
        default=None, 
        description="Start date for tender search (YYYY-MM-DD)",
        example="2024-01-01"
    )
    end_date: Optional[str] = Field(
        default=None, 
        description="End date for tender search (YYYY-MM-DD)",
        example="2024-01-31"
    )
    vectorize: Optional[bool] = Field(
        default=False, 
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
        index_path = os.path.join(OUTPUT_DIR, 'tenders_index.ann')
        metadata_path = os.path.join(OUTPUT_DIR, 'tenders_metadata.json')
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            raise HTTPException(
                status_code=400, 
                detail="Embeddings not found. Run download with --vectorize first."
            )
        similar_tenders = search_similar_tenders(
            query=request.query, 
            top_k=request.top_k
        )
        return [
            TenderSearchResult(
                purchase_number=tender['purchase_number'],
                tender_name=tender['tender_name'],
                similarity_score=tender['similarity_score']
            ) for tender in similar_tenders
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching tenders: {str(e)}")
    
    
if __name__ == "__main__":
    uvicorn.run(
        "tenders_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    ) 