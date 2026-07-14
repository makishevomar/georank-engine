## Architecture Overview
1. **Data & Geospatial Processing:** Raw coordinates are mapped to discrete spatial cells using **Uber H3 Indexing (Resolution 8)**. Feature engineering includes Haversine distances, historical waiting times, and location popularity metrics.
2. **Model Pipeline:** A PyTorch neural network is trained for pointwise ranking, outputting selection probabilities. The preprocessing pipelines (scalers) and model weights are serialized for production use.
3. **Async Serving API:** Built with **FastAPI** to enable low-latency inference (<20ms).
4. **MLOps Monitoring:** Integrated statistical drift detection using the **Kolmogorov-Smirnov (KS) test** to monitor input feature distribution shifts in real-time.

## Project Structure
```text
georank-engine/
├── data/                  # Local datasets
├── models/                # Trained weights and preprocessing pipelines
└── src/
    ├── data_gen.py        # Spatial data simulation & H3 hashing
    ├── pipeline.py        # PyTorch Dataset & DataLoader pipelines
    ├── model.py           # Neural network architecture & training loops
    ├── app.py             # Asynchronous FastAPI application
    └── monitor.py         # MLOps Kolmogorov-Smirnov drift detection

Setup & Execution
1. Install Dependencies
    pip install -r requirements.txt

2. Generate Data & Train Model
    python src/data_gen.py
    python -m src.model

3. Run Async API
    uvicorn src.app:app --reload

Interactive Swagger docs will be available at http://127.0.0.1:8000/docs

4. Run Drift Monitoring
    python src/monitor.py



