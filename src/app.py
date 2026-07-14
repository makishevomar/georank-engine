import torch
import joblib
import h3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.model import PickupPointRanker

app = FastAPI(
    title="GeoRank Engine API",
    description="Asynchronous Geospatial Retrieval & Ranking API for Pickup Points",
    version="1.0.0"
)

try:
    pipeline = joblib.load("models/pipeline.pkl")
    input_dim = len(pipeline.feature_cols)
    
    model = PickupPointRanker(input_dim)
    model.load_state_dict(torch.load("models/ranker_weights.pth"))
    model.eval()
    print("?? Model and Pipeline loaded successfully!")
except Exception as e:
    print(f"?? Failed to load model components. Error: {e}")

class PickupPointCandidate(BaseModel):
    point_id: str
    lat: float
    lng: float
    historical_wait_time_min: float
    popularity_score: float

class RankRequest(BaseModel):
    user_lat: float
    user_lng: float
    candidates: List[PickupPointCandidate]

class RankedPoint(BaseModel):
    point_id: str
    lat: float
    lng: float
    distance_meters: float
    relevance_score: float

class RankResponse(BaseModel):
    user_h3: str
    ranked_points: List[RankedPoint]

@app.post("/api/v1/rank-pickup-points", response_model=RankResponse)
async def rank_pickup_points(request: RankRequest):
    if not request.candidates:
        raise HTTPException(status_code=400, detail="Candidates list cannot be empty")
        
    user_h3 = h3.geo_to_h3(request.user_lat, request.user_lng, resolution=8)
    
    candidates_data = []
    for c in request.candidates:
        dist = h3.point_dist((request.user_lat, request.user_lng), (c.lat, c.lng), unit="m")
        candidates_data.append({
            "point_id": c.point_id,
            "lat": c.lat,
            "lng": c.lng,
            "distance_meters": dist,
            "historical_wait_time_min": c.historical_wait_time_min,
            "popularity_score": c.popularity_score
        })
        
    df_candidates = pd.DataFrame(candidates_data)
    
    try:
        X_scaled = pipeline.transform(df_candidates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature transformation failed: {str(e)}")
        
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = model(X_tensor)
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        
    df_candidates["relevance_score"] = scores
    df_ranked = df_candidates.sort_values(by="relevance_score", ascending=False)
    
    ranked_points = []
    for _, row in df_ranked.iterrows():
        ranked_points.append(
            RankedPoint(
                point_id=row["point_id"],
                lat=row["lat"],
                lng=row["lng"],
                distance_meters=round(row["distance_meters"], 2),
                relevance_score=float(row["relevance_score"])
            )
        )
        
    return RankResponse(user_h3=user_h3, ranked_points=ranked_points)
