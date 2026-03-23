import os
import yaml
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

from src.preprocessing import vectorize_domains
from explainability.explain import get_explainer, explain_domain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="DGA Detection Engine",
    description="Enterprise API for Deep Learning DGA Classification with Explainable AI",
    version="2.0"
)

# Allow CORS for local frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global states
MODEL = None
CONFIG = None
EXPLAINER = None

class PredictRequest(BaseModel):
    domain: str

class CharWeight(BaseModel):
    char: str
    weight: float

class PredictResponse(BaseModel):
    domain: str
    dga_probability: float
    label: str
    explanation: list[CharWeight]
    explanation_text: str


@app.on_event("startup")
async def startup_event():
    global MODEL, CONFIG, EXPLAINER
    logger.info("Starting DGA Engine API...")

    # Load configuration
    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        logger.error(f"Config not found at {cfg_path}")
        return
    with open(cfg_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    # Load trained model
    model_path = CONFIG["evaluation"]["model_save_path"]
    if not Path(model_path).exists():
        logger.error(f"Cannot find trained model at {model_path}. Please run main.py first.")
        return
        
    logger.info(f"Loading Keras model from {model_path}...")
    # Register custom Attention layer explicitly when loading the model
    try:
        from src.model import Attention
        custom_objects = {"Attention": Attention}
        MODEL = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        logger.warning(f"Failed to load model directly, trying with compile=False: {e}")
        try:
            from src.model import Attention
            MODEL = tf.keras.models.load_model(model_path, custom_objects={"Attention": Attention}, compile=False)
        except Exception as e2:
            logger.error(f"Failed to load model even with compile=False: {e2}")
            MODEL = tf.keras.models.load_model(model_path, compile=False)
    
    # Initialize Explainer
    EXPLAINER = get_explainer()
    logger.info("DGA Engine Ready!")


@app.post("/api/predict", response_model=PredictResponse)
async def predict_domain(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    domain = request.domain.strip().lower()
    if not domain:
        raise HTTPException(status_code=400, detail="Domain cannot be empty.")
        
    max_len = CONFIG["model"]["max_sequence_length"]
    
    # Generate Explanation First (this also computes predictions internally)
    raw_explanation = explain_domain(
        domain=domain,
        model=MODEL,
        explainer=EXPLAINER,
        max_len=max_len,
        num_features=max_len
    )
    
    # Raw Prediction
    import pandas as pd
    padded_X = vectorize_domains(pd.Series([domain]), max_len=max_len)
    prob = float(MODEL.predict(padded_X, verbose=0)[0][0])
    
    if prob >= 0.60:
        label = "DGA"
    elif prob >= 0.40:
        label = "Suspicious"
    else:
        label = "Legit"
        
    explanation = [CharWeight(char=item["char"], weight=item["weight"]) for item in raw_explanation]
    
    # --- V3 Explainability Logic: Dynamic Text Summary ---
    parts = domain.rsplit('.', 2)
    subdomain = parts[0] if len(parts) > 2 else ""
    
    # Analyze weight concentration
    subdomain_weights = sum([item.weight for item in explanation[:len(subdomain)] if item.weight > 0])
    total_positive_weights = sum([item.weight for item in explanation if item.weight > 0])
    subdomain_ratio = subdomain_weights / (total_positive_weights + 1e-9)
    
    has_high_entropy_chars = any(c.isdigit() or c == '-' for c in domain)
    
    if label == "DGA":
        if not has_high_entropy_chars and len(domain) > 10:
            explanation_text = "Dictionary DGA pattern detected (semantic anomaly via forced English word concatenation)."
        elif subdomain_ratio > 0.7:
            explanation_text = "High anomaly detected concentrated in the subdomain, indicating algorithmically generated entropy."
        else:
            explanation_text = "Model predicts DGA based on overall character sequence distribution."
    elif label == "Suspicious":
        explanation_text = "Potential Risk: The domain exhibits some DGA-like traits but retains legitimate structural elements. Manual review recommended."
    else:
        if subdomain_ratio > 0.5 and prob < 0.40:
            explanation_text = "High entropy subdomain detected, but decisively overridden by trusted legitimacy of suffix/root domain."
        else:
            explanation_text = "Trusted suffix/root domain gives strong evidence of legitimacy."

    return PredictResponse(
        domain=domain,
        dga_probability=prob,
        label=label,
        explanation=explanation,
        explanation_text=explanation_text
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
