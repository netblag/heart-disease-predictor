"""
Heart Disease Prediction - Model Training & Evaluation
======================================================
Trains, tunes, and evaluates 5 ML models with comprehensive metrics.
Implements cross-validation, hyperparameter tuning, and ensemble methods.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# Model configurations with hyperparameter grids
MODEL_CONFIGS = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(
            random_state=42, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
        ),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 1.0]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto']
        }
    }
}


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'avg_precision': average_precision_score(y_true, y_prob),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> Dict[str, Any]:
    """Perform stratified k-fold cross validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=True,
        n_jobs=-1
    )
    
    return {
        metric.replace('test_', ''): {
            'mean': cv_results[f'test_{metric}'].mean(),
            'std': cv_results[f'test_{metric}'].std(),
            'values': cv_results[f'test_{metric}'].tolist()
        }
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    }


def tune_hyperparameters(model_name: str, config: dict, 
                          X_train, y_train, 
                          n_iter: int = 20) -> Tuple[Any, Dict]:
    """
    Tune hyperparameters using RandomizedSearchCV for efficiency.
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    print(f"  🔧 Tuning {model_name}...", end='', flush=True)
    
    search = RandomizedSearchCV(
        config['model'],
        config['params'],
        n_iter=n_iter,
        cv=5,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    print(f" Best AUC: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_


def train_all_models(datasets: dict, 
                     models_dir: str = 'models/saved',
                     tune: bool = True) -> Dict[str, Any]:
    """
    Train all models with optional hyperparameter tuning.
    Returns comprehensive results dictionary.
    """
    print("\n" + "=" * 60)
    print("🧠 MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    X_train = datasets['X_train']
    X_val = datasets['X_val']
    X_test = datasets['X_test']
    y_train = datasets['y_train']
    y_val = datasets['y_val']
    y_test = datasets['y_test']
    feature_names = datasets['feature_names']
    
    os.makedirs(models_dir, exist_ok=True)
    results = {}
    trained_models = {}
    
    for name, config in MODEL_CONFIGS.items():
        print(f"\n📌 Training: {name}")
        
        # Tune or use default
        if tune:
            model, best_params = tune_hyperparameters(name, config, X_train, y_train)
        else:
            model = config['model']
            model.fit(X_train, y_train)
            best_params = {}
        
        # Cross-validation
        print(f"  📊 Cross-validating...", end='', flush=True)
        cv_scores = cross_validate_model(model, X_train, y_train)
        print(f" CV AUC: {cv_scores['roc_auc']['mean']:.4f} ± {cv_scores['roc_auc']['std']:.4f}")
        
        # Val set predictions
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
        
        # Test set predictions (held out)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        
        # Store results
        results[name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'best_params': best_params,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            },
            'y_test_prob': y_test_prob.tolist(),
            'y_test_pred': y_test_pred.tolist()
        }
        
        trained_models[name] = model
        
        # Save model
        model_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        
        print(f"  ✅ Test AUC: {test_metrics['roc_auc']:.4f} | F1: {test_metrics['f1']:.4f} | Recall: {test_metrics['recall']:.4f}")
    
    # Build ensemble
    print(f"\n📌 Building Voting Ensemble...")
    best_3 = sorted(results.keys(), 
                     key=lambda x: results[x]['test_metrics']['roc_auc'],
                     reverse=True)[:3]
    
    ensemble = VotingClassifier(
        estimators=[(name, trained_models[name]) for name in best_3],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    
    y_ens_pred = ensemble.predict(X_test)
    y_ens_prob = ensemble.predict_proba(X_test)[:, 1]
    ens_metrics = compute_metrics(y_test, y_ens_pred, y_ens_prob)
    fpr_e, tpr_e, thr_e = roc_curve(y_test, y_ens_prob)
    
    results['Ensemble (Top 3)'] = {
        'val_metrics': ens_metrics,
        'test_metrics': ens_metrics,
        'cv_scores': {},
        'best_params': {'components': best_3},
        'roc_curve': {'fpr': fpr_e.tolist(), 'tpr': tpr_e.tolist(), 'thresholds': thr_e.tolist()},
        'y_test_prob': y_ens_prob.tolist(),
        'y_test_pred': y_ens_pred.tolist()
    }
    trained_models['Ensemble (Top 3)'] = ensemble
    joblib.dump(ensemble, os.path.join(models_dir, 'ensemble.pkl'))
    print(f"  ✅ Ensemble AUC: {ens_metrics['roc_auc']:.4f} | F1: {ens_metrics['f1']:.4f}")
    
    # Find best model
    best_model_name = max(
        results.keys(),
        key=lambda x: results[x]['test_metrics']['roc_auc']
    )
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   AUC: {results[best_model_name]['test_metrics']['roc_auc']:.4f}")
    print(f"   F1:  {results[best_model_name]['test_metrics']['f1']:.4f}")
    print(f"   Recall (Sensitivity): {results[best_model_name]['test_metrics']['recall']:.4f}")
    
    # Save results
    results_path = os.path.join(models_dir, 'results.json')
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(results, f, default=convert, indent=2)
    
    # Save best model separately
    joblib.dump(trained_models[best_model_name], 
                os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(datasets['scaler'],
                os.path.join(models_dir, 'scaler.pkl'))
    
    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'model_name': best_model_name,
        'feature_names': datasets['feature_names'],
        'training_date': datetime.now().isoformat(),
        'test_auc': results[best_model_name]['test_metrics']['roc_auc']
    }
    with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results, trained_models, best_model_name


def print_model_comparison(results: Dict):
    """Print formatted model comparison table."""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON TABLE")
    print("=" * 80)
    
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
    print(header)
    print("-" * 80)
    
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['test_metrics']['roc_auc'],
        reverse=True
    )
    
    for name, res in sorted_models:
        m = res['test_metrics']
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")
    
    print("=" * 80)