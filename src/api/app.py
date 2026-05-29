"""
Heart Disease Prediction API
============================
Production-ready REST API for heart disease risk prediction.
Built with FastAPI for high performance and automatic OpenAPI docs.

Endpoints:
- POST /predict      → Single patient prediction
- POST /predict/batch → Batch predictions
- GET  /health       → API health check
- GET  /model/info   → Model metadata and performance
- GET  /features     → Feature descriptions
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Feature engineering imports (same as preprocessing)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── App Configuration ───────────────────────────────────────────────────────

app = FastAPI(
    title="🫀 Heart Disease Risk Predictor API",
    description="""
## Heart Disease Risk Assessment API

This API uses machine learning to predict the risk of heart disease based on 
clinical parameters. It was trained on the UCI Heart Disease dataset with 
multiple algorithms and SHAP-based explainability.

### Key Features
- **Real-time prediction** with confidence scores
- **Risk stratification** (Low / Moderate / High)
- **Batch processing** for multiple patients
- **Clinical feature validation** with medical bounds
- **Model transparency** via feature importance

### ⚠️ Medical Disclaimer
This tool is for research/educational purposes only and should **NOT** be used 
as a substitute for professional medical advice, diagnosis, or treatment.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Data Models ─────────────────────────────────────────────────────────────

class PatientData(BaseModel):
    """Input schema for a single patient's clinical data."""
    
    age: int = Field(..., ge=18, le=100, description="Age in years", example=54)
    sex: int = Field(..., ge=0, le=1, description="Sex: 1=Male, 0=Female", example=1)
    cp: int = Field(..., ge=0, le=3, description="Chest pain type: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic", example=0)
    trestbps: float = Field(..., ge=80, le=250, description="Resting blood pressure (mm Hg)", example=130)
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)", example=250)
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dl: 1=True, 0=False", example=0)
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy", example=1)
    thalach: float = Field(..., ge=60, le=250, description="Maximum heart rate achieved", example=150)
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina: 1=Yes, 0=No", example=0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise", example=1.5)
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping", example=1)
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)", example=0)
    thal: int = Field(..., ge=0, le=3, description="Thalassemia: 0=Unknown, 1=Normal, 2=Fixed defect, 3=Reversible defect", example=2)
    
    @validator('thalach')
    def validate_max_hr(cls, v, values):
        if 'age' in values:
            expected_max = 220 - values['age']
            if v > expected_max * 1.15:
                raise ValueError(f"Max heart rate {v} seems too high for age {values['age']}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
                "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    prediction: int
    probability: float
    risk_level: str
    risk_description: str
    confidence: str
    key_risk_factors: List[str]
    recommendations: List[str]
    model_version: str
    timestamp: str


class BatchRequest(BaseModel):
    patients: List[PatientData]


class ModelInfo(BaseModel):
    model_name: str
    version: str
    training_date: str
    test_auc: float
    feature_count: int
    supported_features: List[str]


# ─── Model Loading ────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models/saved')

_model = None
_scaler = None
_metadata = None
_feature_names = None


def load_model_artifacts():
    """Load model, scaler and metadata from disk."""
    global _model, _scaler, _metadata, _feature_names
    
    try:
        model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        meta_path = os.path.join(MODELS_DIR, 'metadata.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run main.py first.")
        
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        
        with open(meta_path) as f:
            _metadata = json.load(f)
        
        _feature_names = _metadata['feature_names']
        print(f"✅ Model loaded: {_metadata['model_name']} (AUC: {_metadata.get('test_auc', 'N/A'):.4f})")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("   API will run in demo mode")


@app.on_event("startup")
async def startup_event():
    load_model_artifacts()


# ─── Feature Engineering (mirror of preprocess.py) ───────────────────────────

def engineer_patient_features(patient: PatientData) -> dict:
    """Apply same feature engineering as training pipeline."""
    d = patient.model_dump()
    
    d['hr_age_ratio'] = d['thalach'] / d['age']
    d['expected_max_hr'] = 220 - d['age']
    d['hr_reserve_pct'] = (d['thalach'] / d['expected_max_hr']) * 100
    d['chol_age_interaction'] = d['chol'] * d['age'] / 1000
    d['hypertension_risk'] = int(d['trestbps'] >= 140 or d['trestbps'] >= 130)
    d['st_depression_severe'] = int(d['oldpeak'] >= 2.0)
    d['multi_vessel'] = int(d['ca'] >= 2)
    d['clinical_risk_score'] = (
        int(d['age'] > 55) * 2 +
        d['sex'] * 1.5 +
        int(d['cp'] == 0) * 2 +
        int(d['trestbps'] > 140) * 1 +
        int(d['chol'] > 240) * 1 +
        d['exang'] * 2 +
        int(d['oldpeak'] > 1) * 1.5 +
        d['ca'] * 1.5
    )
    d['age_group'] = min(int(d['age'] // 10) - 3, 4)
    d['cp_risk'] = {0: 3, 1: 2, 2: 1, 3: 0}.get(d['cp'], 0)
    
    return d


def get_risk_factors(patient: PatientData) -> List[str]:
    """Identify key risk factors for the patient."""
    factors = []
    d = patient.model_dump()
    
    if d['age'] > 55: factors.append(f"Age over 55 ({d['age']} years)")
    if d['sex'] == 1: factors.append("Male sex (higher baseline risk)")
    if d['cp'] == 0: factors.append("Asymptomatic chest pain pattern")
    if d['trestbps'] > 140: factors.append(f"Hypertension (BP: {d['trestbps']} mm Hg)")
    if d['chol'] > 240: factors.append(f"High cholesterol ({d['chol']} mg/dl)")
    if d['exang'] == 1: factors.append("Exercise-induced angina present")
    if d['oldpeak'] > 2.0: factors.append(f"Significant ST depression ({d['oldpeak']})")
    if d['ca'] >= 2: factors.append(f"Multiple vessel disease ({d['ca']} vessels)")
    if d['thal'] == 3: factors.append("Reversible thalassemia defect")
    
    return factors if factors else ["No major individual risk factors identified"]


def get_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
    """Generate clinical recommendations based on risk level."""
    base_recs = [
        "Consult a cardiologist for comprehensive evaluation",
        "Maintain a heart-healthy diet (Mediterranean diet recommended)",
        "Regular aerobic exercise (150 min/week moderate intensity)"
    ]
    
    if risk_level == 'HIGH':
        return [
            "🚨 Immediate cardiology consultation strongly recommended",
            "Consider stress testing and coronary imaging",
            "Review current medications with physician",
            "Lifestyle modifications: smoking cessation, diet, exercise",
            "Monitor blood pressure and cholesterol regularly"
        ]
    elif risk_level == 'MODERATE':
        return [
            "⚠️ Schedule cardiology appointment within 1-2 weeks",
            "Increase physical activity gradually under medical supervision",
            "Optimize modifiable risk factors (BP, cholesterol, weight)",
            "Regular monitoring every 3-6 months",
        ] + base_recs[:1]
    else:
        return [
            "✅ Continue regular preventive care",
            "Annual cardiovascular check-up recommended",
        ] + base_recs


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "🫀 Heart Disease Risk Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "timestamp": datetime.now().isoformat(),
        "model_name": _metadata.get('model_name', 'unknown') if _metadata else 'not loaded'
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Return model metadata and training information."""
    if not _metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=_metadata.get('model_name', 'Unknown'),
        version="1.0.0",
        training_date=_metadata.get('training_date', 'Unknown'),
        test_auc=_metadata.get('test_auc', 0.0),
        feature_count=len(_feature_names) if _feature_names else 0,
        supported_features=_feature_names or []
    )


@app.get("/features", tags=["Model"])
async def get_features():
    """Return descriptions of all features."""
    return {
        "input_features": {
            "age": "Age in years (18-100)",
            "sex": "Biological sex: 1=Male, 0=Female",
            "cp": "Chest pain type: 0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic",
            "trestbps": "Resting blood pressure in mm Hg",
            "chol": "Serum cholesterol in mg/dl",
            "fbs": "Fasting blood sugar >120 mg/dl: 1=Yes, 0=No",
            "restecg": "Resting ECG: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy",
            "thalach": "Maximum heart rate achieved during exercise",
            "exang": "Exercise-induced angina: 1=Yes, 0=No",
            "oldpeak": "ST depression induced by exercise relative to rest (0-10)",
            "slope": "Slope of peak exercise ST: 0=Upsloping, 1=Flat, 2=Downsloping",
            "ca": "Number of major vessels colored by fluoroscopy (0-4)",
            "thal": "Thalassemia: 0=Unknown, 1=Normal, 2=Fixed defect, 3=Reversible defect"
        },
        "engineered_features": [
            "hr_age_ratio", "hr_reserve_pct", "chol_age_interaction",
            "hypertension_risk", "st_depression_severe", "multi_vessel",
            "clinical_risk_score", "age_group", "cp_risk"
        ]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(patient: PatientData):
    """
    Predict heart disease risk for a single patient.
    
    Returns probability score, risk level, key risk factors, and recommendations.
    """
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training pipeline first."
        )
    
    try:
        # Engineer features
        patient_dict = engineer_patient_features(patient)
        
        # Build feature vector in correct order
        X = np.array([[patient_dict[f] for f in _feature_names]])
        X_scaled = _scaler.transform(X)
        
        # Predict
        prob = float(_model.predict_proba(X_scaled)[0][1])
        pred = int(prob >= 0.5)
        
        # Risk stratification
        if prob >= 0.70:
            risk_level = "HIGH"
            risk_desc = "High risk of heart disease. Immediate medical attention recommended."
        elif prob >= 0.40:
            risk_level = "MODERATE"
            risk_desc = "Moderate risk. Medical evaluation and lifestyle changes advised."
        else:
            risk_level = "LOW"
            risk_desc = "Low risk based on current parameters. Maintain healthy lifestyle."
        
        # Confidence
        margin = abs(prob - 0.5)
        if margin >= 0.35: confidence = "Very High"
        elif margin >= 0.2: confidence = "High"
        elif margin >= 0.1: confidence = "Moderate"
        else: confidence = "Low"
        
        risk_factors = get_risk_factors(patient)
        recommendations = get_recommendations(risk_level, risk_factors)
        
        return PredictionResponse(
            prediction=pred,
            probability=round(prob, 4),
            risk_level=risk_level,
            risk_description=risk_desc,
            confidence=confidence,
            key_risk_factors=risk_factors,
            recommendations=recommendations,
            model_version=_metadata.get('model_name', '1.0.0') if _metadata else '1.0.0',
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Predict heart disease risk for multiple patients at once.
    Maximum 100 patients per request.
    """
    if len(request.patients) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 patients per batch request")
    
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for i, patient in enumerate(request.patients):
        try:
            patient_dict = engineer_patient_features(patient)
            X = np.array([[patient_dict[f] for f in _feature_names]])
            X_scaled = _scaler.transform(X)
            prob = float(_model.predict_proba(X_scaled)[0][1])
            pred = int(prob >= 0.5)
            risk = "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.4 else "LOW"
            
            results.append({
                "patient_index": i,
                "prediction": pred,
                "probability": round(prob, 4),
                "risk_level": risk
            })
        except Exception as e:
            results.append({"patient_index": i, "error": str(e)})
    
    high_risk = sum(1 for r in results if r.get('risk_level') == 'HIGH')
    moderate_risk = sum(1 for r in results if r.get('risk_level') == 'MODERATE')
    low_risk = sum(1 for r in results if r.get('risk_level') == 'LOW')
    
    return {
        "total_patients": len(request.patients),
        "summary": {
            "high_risk": high_risk,
            "moderate_risk": moderate_risk,
            "low_risk": low_risk,
            "disease_predicted": sum(1 for r in results if r.get('prediction') == 1)
        },
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)