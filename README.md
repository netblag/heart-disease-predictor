# 🫀 Heart Disease Risk Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi)
![Tests](https://img.shields.io/badge/Tests-32%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A production-grade machine learning system for cardiovascular disease risk assessment.**  
Trains 5 models + ensemble, explains predictions with SHAP, and serves results via REST API.

[Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [API](#api-usage) · [Results](#model-results) · [فارسی](https://github.com/netblag/heart-disease-predictor/blob/main/README.fa.md)

</div>

---

## ⚠️ Medical Disclaimer

> This project is for **educational and research purposes only**. It must **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Features

- **Multi-model training**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM + Soft Voting Ensemble
- **Hyperparameter tuning**: RandomizedSearchCV with stratified k-fold cross-validation
- **Class imbalance handling**: SMOTE oversampling on training set only
- **Feature engineering**: 9 clinically-informed derived features (HR reserve, composite risk score, etc.)
- **Explainability**: SHAP values for both global importance and per-patient explanations
- **Production API**: FastAPI with Pydantic validation, batch endpoints, OpenAPI docs
- **Visualizations**: EDA dashboard, ROC comparison, confusion matrix grid, SHAP plots
- **32 automated tests** covering preprocessing, models, metrics, and API validation
- **Docker support** for containerized deployment

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | **AUC** |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.891 | 0.891 | 0.891 | **0.963** |
| Ensemble (Top 3) | 0.875 | 0.870 | 0.909 | 0.889 | 0.960 |
| SVM | 0.860 | 0.860 | 0.891 | 0.875 | 0.954 |
| Random Forest | 0.850 | 0.851 | 0.882 | 0.866 | 0.951 |
| LightGBM | 0.895 | 0.916 | 0.891 | 0.903 | 0.947 |
| XGBoost | 0.875 | 0.883 | 0.891 | 0.887 | 0.947 |

Cross-validation AUC: **0.948 ± 0.013** (5-fold stratified)

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
# Full pipeline with hyperparameter tuning (recommended, ~5–10 min)
python main.py

# Fast mode — no tuning, good for testing (~30 seconds)
python main.py --no-tune
```

This will:
1. Generate the dataset (UCI-based synthetic data, 1000 patients)
2. Preprocess and engineer features
3. Train and evaluate all 6 models
4. Generate SHAP explanations
5. Save all plots to `reports/figures/`
6. Save trained models to `models/saved/`

### 5. Start the API server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Architecture

```
heart-disease-predictor/
│
├── main.py                        # Pipeline orchestrator
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   │   └── preprocess.py          # Cleaning, feature engineering, splits, SMOTE
│   ├── models/
│   │   ├── train.py               # Model configs, tuning, CV, evaluation, ensemble
│   │   └── explain.py             # SHAP values and explainability plots
│   ├── visualization/
│   │   └── plots.py               # EDA, ROC curves, confusion matrices
│   └── api/
│       └── app.py                 # FastAPI app with prediction endpoints
│
├── tests/
│   └── test_pipeline.py           # 32 pytest tests
│
├── data/
│   ├── raw/                       # Raw CSV (generated on first run)
│   └── processed/                 # Feature-engineered CSV
│
├── models/
│   └── saved/                     # Serialized models, scaler, metadata
│
├── reports/
│   └── figures/                   # All generated plots (PNG)
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Feature Engineering

Beyond the 13 original UCI features, 9 clinically-informed features are derived:

| Feature | Description | Medical Basis |
|---|---|---|
| `hr_age_ratio` | Max HR ÷ Age | Age-adjusted heart rate capacity |
| `expected_max_hr` | 220 − Age | Standard cardiology formula |
| `hr_reserve_pct` | (Max HR / Expected) × 100 | Percentage of HR reserve used |
| `chol_age_interaction` | Cholesterol × Age / 1000 | Compound cardiovascular risk |
| `hypertension_risk` | BP ≥ 130 mm Hg | ACC/AHA hypertension threshold |
| `st_depression_severe` | ST depression ≥ 2.0 | Significant ischemia marker |
| `multi_vessel` | # vessels ≥ 2 | Multi-vessel coronary disease |
| `clinical_risk_score` | Weighted composite of 8 factors | Clinical risk stratification |
| `cp_risk` | Remapped chest pain (0=lowest risk) | Asymptomatic = highest risk |

---

## API Usage

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60,
    "sex": 1,
    "cp": 0,
    "trestbps": 150,
    "chol": 280,
    "fbs": 0,
    "restecg": 1,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.8734,
  "risk_level": "HIGH",
  "risk_description": "High risk of heart disease. Immediate medical attention recommended.",
  "confidence": "Very High",
  "key_risk_factors": [
    "Age over 55 (60 years)",
    "Male sex (higher baseline risk)",
    "Asymptomatic chest pain pattern",
    "Hypertension (BP: 150 mm Hg)",
    "Exercise-induced angina present",
    "Significant ST depression (2.5)",
    "Multiple vessel disease (2 vessels)",
    "Reversible thalassemia defect"
  ],
  "recommendations": [
    "🚨 Immediate cardiology consultation strongly recommended",
    "Consider stress testing and coronary imaging",
    "Review current medications with physician",
    "Lifestyle modifications: smoking cessation, diet, exercise",
    "Monitor blood pressure and cholesterol regularly"
  ],
  "model_version": "Logistic Regression",
  "timestamp": "2026-05-29T12:00:00"
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"patients": [ {...}, {...} ]}'
```

### Other endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/model/info` | Model metadata |
| GET | `/features` | Feature descriptions |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch (up to 100) |

---

## Docker Deployment

```bash
# Build and run
cd docker
docker-compose up --build

# API will be available at http://localhost:8000
```

---

## Generated Outputs

After running the pipeline, the following files are created:

```
reports/figures/
├── eda_overview.png          # 8-panel EDA dashboard
├── roc_comparison.png        # ROC curves + metrics bar chart
├── confusion_matrices.png    # Confusion matrix grid for all models
├── shap_feature_importance.png  # Top 15 features by mean |SHAP|
└── shap_summary.png          # SHAP beeswarm plot

models/saved/
├── best_model.pkl            # Serialized best model
├── scaler.pkl                # Fitted StandardScaler
├── ensemble.pkl              # Voting ensemble
├── results.json              # All metrics for all models
└── metadata.json             # Training metadata
```

---

## Dataset

Based on the **UCI Heart Disease Dataset** (Cleveland Clinic Foundation).

- **Original source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **13 clinical features** + 1 target (binary: disease / no disease)
- **1000 patients** (synthetic generation preserving UCI statistical distributions)
- **~54% disease prevalence** (matching the original dataset)

---


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Make sure all tests pass before submitting.

```bash
pytest tests/ -v --tb=short
```
