from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib
import os
from typing import Optional, Literal
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Prediction Service", version="1.0.0")

# CORS - restrictive by default, Node will handle browser CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Global model storage
models = {}

# Configuration
MODEL_RF_PATH = os.getenv("MODEL_RF_PATH", "runs/rf/model.joblib")
MODEL_MLP_PATH = os.getenv("MODEL_MLP_PATH", "runs/mlp/model.joblib")
MODEL_MLP_FALLBACK_PATH = os.getenv("MODEL_MLP_FALLBACK_PATH", "runs/neural/model.joblib")
MAX_SMILES_LENGTH = int(os.getenv("MAX_SMILES_LENGTH", "500"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "10000"))

# Valid amino acids
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


class PredictionRequest(BaseModel):
    compound_iso_smiles: str = Field(..., min_length=1, description="SMILES string")
    target_sequence: str = Field(..., min_length=1, description="Protein sequence")
    model: Optional[Literal["rf", "mlp"]] = Field("rf", description="Model to use")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")

    @validator('compound_iso_smiles')
    def validate_smiles(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("SMILES cannot be empty")
        if len(v) > MAX_SMILES_LENGTH:
            raise ValueError(f"SMILES too long (max {MAX_SMILES_LENGTH} characters)")
        return v

    @validator('target_sequence')
    def validate_sequence(cls, v):
        v = v.strip().upper()
        if not v:
            raise ValueError("Protein sequence cannot be empty")
        if len(v) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Sequence too long (max {MAX_SEQUENCE_LENGTH} characters)")

        # Check for invalid amino acids
        invalid_chars = set(v) - VALID_AA
        if invalid_chars:
            raise ValueError(f"Invalid amino acids found: {', '.join(sorted(invalid_chars))}")

        return v


class PredictionResponse(BaseModel):
    probability: float
    label: int
    model_used: str
    errors: Optional[list[str]] = None


def _load_model(key: str, path: str) -> bool:
    if os.path.exists(path):
        try:
            models[key] = joblib.load(path)
            logger.info("Loaded %s model from %s", key, path)
            return True
        except Exception as exc:
            logger.error("Failed to load %s model from %s: %s", key, path, exc)
            return False
    logger.warning("%s model not found at %s", key, path)
    return False


@app.on_event("startup")
async def load_models():
    """Load models at startup"""
    logger.info("Loading models...")

    _load_model("rf", MODEL_RF_PATH)

    if not _load_model("mlp", MODEL_MLP_PATH):
        if os.path.exists(MODEL_MLP_FALLBACK_PATH):
            _load_model("mlp", MODEL_MLP_FALLBACK_PATH)
        else:
            logger.info("MLP model not found at %s (optional)", MODEL_MLP_PATH)

    if not models:
        logger.error("No models loaded! Service will not work properly.")
    else:
        logger.info("Service ready with models: %s", list(models.keys()))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "model_paths": {
            "rf": MODEL_RF_PATH,
            "mlp": MODEL_MLP_PATH
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction using the specified model
    """
    errors = []

    available = list(models.keys())
    if request.model not in models:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Model '{request.model}' not available",
                "available_models": available
            }
        )

    try:
        model = models[request.model]

        input_data = pd.DataFrame([
            {
                "compound_iso_smiles": request.compound_iso_smiles,
                "target_sequence": request.target_sequence
            }
        ])

        logger.info("Prediction input shape: %s", input_data.shape)

        if not hasattr(model, "predict_proba"):
            logger.error("Model '%s' does not support predict_proba", request.model)
            raise HTTPException(status_code=500, detail="Model does not support predict_proba")

        probability = float(model.predict_proba(input_data)[0, 1])
        label = int(probability >= request.threshold)

        logger.info(
            "Prediction made: model=%s prob=%.6f label=%s threshold=%s",
            request.model,
            probability,
            label,
            request.threshold
        )

        return PredictionResponse(
            probability=probability,
            label=label,
            model_used=request.model,
            errors=errors if errors else None
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
