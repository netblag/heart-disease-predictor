"""
Heart Disease Predictor - Main Pipeline
=======================================
Orchestrates the complete ML pipeline:
1. Generate / download dataset
2. Preprocess & feature engineering
3. Train & evaluate models
4. Generate SHAP explanations
5. Create visualizations
6. Save artifacts for API serving

Usage:
    python main.py              # Full pipeline
    python main.py --no-tune   # Skip hyperparameter tuning (faster)
    python main.py --api-only  # Only start the API server
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocess import run_pipeline
from src.models.train import train_all_models, print_model_comparison
from src.models.explain import run_explainability_pipeline
from src.visualization.plots import run_visualization_pipeline


def generate_uci_dataset(output_path: str):
    """
    Generate a high-fidelity synthetic dataset based on UCI Heart Disease statistics.
    
    This ensures the pipeline works without needing to download data manually.
    Uses validated statistical distributions from the UCI Cleveland dataset.
    """
    np.random.seed(42)
    n = 1000  # Larger than original for better model training
    
    print("📥 Generating UCI-based Heart Disease Dataset...")
    
    # Disease prevalence ~54% (from UCI)
    disease = np.random.binomial(1, 0.54, n)
    
    rows = []
    for has_disease in disease:
        if has_disease:
            age = np.clip(np.random.normal(56, 9), 30, 77)
            sex = np.random.choice([0, 1], p=[0.25, 0.75])
            cp = np.random.choice([0, 1, 2, 3], p=[0.40, 0.30, 0.20, 0.10])
            trestbps = np.clip(np.random.normal(135, 18), 94, 200)
            chol = np.clip(np.random.normal(251, 55), 126, 417)
            fbs = np.random.choice([0, 1], p=[0.82, 0.18])
            restecg = np.random.choice([0, 1, 2], p=[0.45, 0.50, 0.05])
            thalach = np.clip(np.random.normal(139, 23), 71, 202)
            exang = np.random.choice([0, 1], p=[0.32, 0.68])
            oldpeak = np.clip(np.random.exponential(1.6), 0, 6.2)
            slope = np.random.choice([0, 1, 2], p=[0.30, 0.55, 0.15])
            ca = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15])
            thal = np.random.choice([0, 1, 2, 3], p=[0.05, 0.20, 0.15, 0.60])
        else:
            age = np.clip(np.random.normal(52, 10), 29, 77)
            sex = np.random.choice([0, 1], p=[0.45, 0.55])
            cp = np.random.choice([0, 1, 2, 3], p=[0.10, 0.25, 0.40, 0.25])
            trestbps = np.clip(np.random.normal(128, 17), 94, 180)
            chol = np.clip(np.random.normal(242, 50), 141, 417)
            fbs = np.random.choice([0, 1], p=[0.87, 0.13])
            restecg = np.random.choice([0, 1, 2], p=[0.65, 0.30, 0.05])
            thalach = np.clip(np.random.normal(158, 19), 96, 202)
            exang = np.random.choice([0, 1], p=[0.83, 0.17])
            oldpeak = np.clip(np.random.exponential(0.8), 0, 4.2)
            slope = np.random.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
            ca = np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03])
            thal = np.random.choice([0, 1, 2, 3], p=[0.05, 0.55, 0.30, 0.10])
        
        rows.append([
            int(age), int(sex), int(cp), round(trestbps, 1), round(chol, 1),
            int(fbs), int(restecg), round(thalach, 1), int(exang), 
            round(oldpeak, 2), int(slope), int(ca), int(thal), int(has_disease)
        ])
    
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, header=False)
    
    print(f"  ✅ Generated {len(df)} patient records")
    print(f"  📁 Saved to: {output_path}")
    print(f"  🎯 Disease prevalence: {df['target'].mean():.1%}")
    return output_path


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🫀  HEART DISEASE RISK PREDICTOR                          ║
║        Machine Learning Pipeline v1.0                        ║
║                                                              ║
║    Models: LR | RF | XGBoost | LightGBM | SVM | Ensemble    ║
║    Analysis: SHAP Explainability + Clinical Insights         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_full_pipeline(tune: bool = True):
    """Execute the complete ML pipeline."""
    
    print_banner()
    start_time = time.time()
    
    # ── Paths ─────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data/raw/heart_disease.csv')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models/saved')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports/figures')
    
    # ── Step 1: Data ───────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        generate_uci_dataset(DATA_PATH)
    
    # ── Step 2: Preprocess ─────────────────────────────────────────
    datasets, df_processed = run_pipeline(DATA_PATH, PROCESSED_DIR)
    
    # ── Step 3: Train Models ───────────────────────────────────────
    results, trained_models, best_model_name = train_all_models(
        datasets, models_dir=MODELS_DIR, tune=tune
    )
    print_model_comparison(results)
    
    # ── Step 4: Visualizations ─────────────────────────────────────
    run_visualization_pipeline(
        df_processed, results, datasets['y_test'], REPORTS_DIR
    )
    
    # ── Step 5: SHAP Explainability ────────────────────────────────
    run_explainability_pipeline(
        trained_models, datasets, best_model_name, REPORTS_DIR
    )
    
    # ── Step 6: Final Summary ──────────────────────────────────────
    elapsed = time.time() - start_time
    best_auc = results[best_model_name]['test_metrics']['roc_auc']
    best_f1  = results[best_model_name]['test_metrics']['f1']
    best_rec = results[best_model_name]['test_metrics']['recall']
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  ⏱️  Total time    : {elapsed:.1f}s")
    print(f"  🏆 Best model    : {best_model_name}")
    print(f"  📊 Test AUC      : {best_auc:.4f}")
    print(f"  📊 Test F1       : {best_f1:.4f}")
    print(f"  📊 Sensitivity   : {best_rec:.4f}")
    print(f"\n  📁 Artifacts saved:")
    print(f"     Models  → {MODELS_DIR}/")
    print(f"     Plots   → {REPORTS_DIR}/")
    print(f"\n  🚀 To start API server:")
    print(f"     uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload")
    print(f"     Then visit: http://localhost:8000/docs")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Heart Disease Prediction Pipeline'
    )
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip hyperparameter tuning (faster, for testing)')
    parser.add_argument('--api-only', action='store_true',
                        help='Start API server only (requires trained model)')
    
    args = parser.parse_args()
    
    if args.api_only:
        import uvicorn
        print("🚀 Starting API server...")
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_full_pipeline(tune=not args.no_tune)


if __name__ == '__main__':
    main()