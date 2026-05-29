"""
Heart Disease Predictor - Test Suite
=====================================
Comprehensive tests for data preprocessing, model training,
feature engineering, and API endpoints.

Run with: pytest tests/ -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import json
import tempfile

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import clean_data, engineer_features, prepare_datasets


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_data():
    """Dataset large enough for stratified splitting (50+ samples)."""
    np.random.seed(42)
    n = 80
    data = {
        'age': np.random.randint(35, 75, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.randint(100, 180, n),
        'chol': np.random.randint(150, 350, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(90, 200, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.round(np.random.uniform(0, 4, n), 1),
        'slope': np.random.randint(0, 3, n),
        'ca': np.random.randint(0, 4, n),
        'thal': np.random.randint(0, 4, n),
        'target': np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def engineered_data(sample_raw_data):
    """Feature-engineered dataset."""
    cleaned = clean_data(sample_raw_data)
    return engineer_features(cleaned)


# ─── Data Preprocessing Tests ─────────────────────────────────────────────────

class TestDataCleaning:
    
    def test_clean_data_no_null_values(self, sample_raw_data):
        """No nulls should remain after cleaning."""
        cleaned = clean_data(sample_raw_data)
        assert cleaned.isnull().sum().sum() == 0
    
    def test_clean_data_target_binary(self, sample_raw_data):
        """Target must be binary (0 or 1)."""
        cleaned = clean_data(sample_raw_data)
        assert set(cleaned['target'].unique()).issubset({0, 1})
    
    def test_clean_data_removes_zero_bp(self):
        """Rows with zero blood pressure should be removed."""
        np.random.seed(0)
        n = 20
        rows = {
            'age': np.random.randint(35,70,n), 'sex': np.ones(n,int),
            'cp': np.zeros(n,int), 'trestbps': np.random.randint(110,160,n),
            'chol': np.random.randint(180,300,n), 'fbs': np.zeros(n,int),
            'restecg': np.zeros(n,int), 'thalach': np.random.randint(120,180,n),
            'exang': np.zeros(n,int), 'oldpeak': np.ones(n)*0.5,
            'slope': np.ones(n,int), 'ca': np.zeros(n,int),
            'thal': np.ones(n,int)*2, 'target': np.random.randint(0,2,n),
        }
        df = pd.DataFrame(rows)
        df.loc[0, 'trestbps'] = 0   # This row should be removed
        df.loc[1, 'chol'] = 0       # This row should be removed
        cleaned = clean_data(df)
        assert len(cleaned) == n - 2
    
    def test_clean_data_handles_question_marks(self):
        """UCI dataset uses '?' for missing values — should be imputed."""
        rows = []
        for i in range(20):
            rows.append({
                'age': 45+i, 'sex': i%2, 'cp': i%4,
                'trestbps': 120+i, 'chol': 200+i*2, 'fbs': 0,
                'restecg': i%3, 'thalach': 140+i, 'exang': 0,
                'oldpeak': 0.5, 'slope': 1, 'ca': '?' if i < 3 else i%4,
                'thal': '?' if i < 3 else (i%3)+1, 'target': i%2
            })
        df = pd.DataFrame(rows)
        cleaned = clean_data(df)
        assert cleaned['ca'].isnull().sum() == 0
        assert cleaned['thal'].isnull().sum() == 0
    
    def test_clean_data_preserves_valid_records(self, sample_raw_data):
        """Cleaning should preserve all valid records."""
        cleaned = clean_data(sample_raw_data)
        assert len(cleaned) == len(sample_raw_data)
    
    def test_clean_data_correct_dtypes(self, sample_raw_data):
        """All columns should be numeric after cleaning."""
        cleaned = clean_data(sample_raw_data)
        for col in cleaned.columns:
            assert pd.api.types.is_numeric_dtype(cleaned[col]), \
                f"Column '{col}' is not numeric"


class TestFeatureEngineering:
    
    def test_new_features_created(self, sample_raw_data):
        """Feature engineering should add new columns."""
        cleaned = clean_data(sample_raw_data)
        original_cols = set(cleaned.columns)
        engineered = engineer_features(cleaned)
        new_cols = set(engineered.columns) - original_cols
        assert len(new_cols) >= 7, f"Expected ≥7 new features, got {len(new_cols)}"
    
    def test_hr_age_ratio_computed_correctly(self, sample_raw_data):
        """hr_age_ratio = thalach / age."""
        cleaned = clean_data(sample_raw_data)
        engineered = engineer_features(cleaned)
        expected = engineered['thalach'] / engineered['age']
        np.testing.assert_allclose(engineered['hr_age_ratio'], expected)
    
    def test_expected_max_hr_formula(self, sample_raw_data):
        """expected_max_hr = 220 - age (standard cardiology formula)."""
        cleaned = clean_data(sample_raw_data)
        engineered = engineer_features(cleaned)
        expected = 220 - engineered['age']
        np.testing.assert_array_equal(engineered['expected_max_hr'], expected)
    
    def test_clinical_risk_score_non_negative(self, sample_raw_data):
        """Clinical risk score should always be non-negative."""
        cleaned = clean_data(sample_raw_data)
        engineered = engineer_features(cleaned)
        assert (engineered['clinical_risk_score'] >= 0).all()
    
    def test_binary_features_are_binary(self, sample_raw_data):
        """Binary engineered features should only contain 0/1."""
        cleaned = clean_data(sample_raw_data)
        engineered = engineer_features(cleaned)
        for col in ['hypertension_risk', 'st_depression_severe', 'multi_vessel']:
            assert set(engineered[col].unique()).issubset({0, 1}), \
                f"Feature '{col}' has non-binary values"
    
    def test_cp_risk_mapping(self, sample_raw_data):
        """Asymptomatic cp (0) should map to highest risk (3)."""
        cleaned = clean_data(sample_raw_data)
        engineered = engineer_features(cleaned)
        mask = engineered['cp'] == 0
        assert (engineered.loc[mask, 'cp_risk'] == 3).all()
    
    def test_original_features_preserved(self, sample_raw_data):
        """Feature engineering must not modify original features."""
        cleaned = clean_data(sample_raw_data)
        original_values = cleaned['age'].copy()
        engineered = engineer_features(cleaned)
        pd.testing.assert_series_equal(engineered['age'], original_values)


class TestDatasetPreparation:
    
    def test_train_val_test_split_sizes(self, engineered_data):
        """Check that split sizes are approximately correct."""
        n = len(engineered_data)
        datasets = prepare_datasets(engineered_data, test_size=0.2, val_size=0.1, apply_smote=False)
        
        total = len(datasets['y_train']) + len(datasets['y_val']) + len(datasets['y_test'])
        # After SMOTE total can change; check test and val
        assert len(datasets['y_test']) == pytest.approx(n * 0.2, abs=2)
    
    def test_no_target_in_features(self, engineered_data):
        """Target column must not appear in feature matrix."""
        datasets = prepare_datasets(engineered_data, apply_smote=False)
        assert 'target' not in datasets['feature_names']
    
    def test_scaler_fitted(self, engineered_data):
        """Scaler should be fitted (has mean_ attribute)."""
        datasets = prepare_datasets(engineered_data, apply_smote=False)
        assert hasattr(datasets['scaler'], 'mean_')
    
    def test_x_train_shape_matches_features(self, engineered_data):
        """X_train columns should match feature_names count."""
        datasets = prepare_datasets(engineered_data, apply_smote=False)
        assert datasets['X_train'].shape[1] == len(datasets['feature_names'])
    
    def test_stratified_split(self, engineered_data):
        """Both classes should be present in test set."""
        datasets = prepare_datasets(engineered_data, apply_smote=False)
        unique_classes = np.unique(datasets['y_test'])
        assert len(unique_classes) == 2, "Test set should have both classes"


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestModelPipeline:
    
    @pytest.fixture
    def small_datasets(self, engineered_data):
        """Prepare small dataset for fast model testing."""
        return prepare_datasets(engineered_data, apply_smote=False)
    
    def test_logistic_regression_trains(self, small_datasets):
        """Logistic Regression should train and predict."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(small_datasets['X_train'], small_datasets['y_train'])
        preds = model.predict(small_datasets['X_test'])
        assert len(preds) == len(small_datasets['y_test'])
    
    def test_random_forest_trains(self, small_datasets):
        """Random Forest should train and output probabilities."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(small_datasets['X_train'], small_datasets['y_train'])
        probs = model.predict_proba(small_datasets['X_test'])
        assert probs.shape == (len(small_datasets['y_test']), 2)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_xgboost_trains(self, small_datasets):
        """XGBoost should train without errors."""
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, 
                                    eval_metric='logloss', use_label_encoder=False)
        model.fit(small_datasets['X_train'], small_datasets['y_train'])
        probs = model.predict_proba(small_datasets['X_test'])[:, 1]
        assert all(0 <= p <= 1 for p in probs)
    
    def test_model_auc_above_random(self, small_datasets):
        """Any trained model should beat random classifier (AUC > 0.5)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(small_datasets['X_train'], small_datasets['y_train'])
        probs = model.predict_proba(small_datasets['X_test'])[:, 1]
        auc = roc_auc_score(small_datasets['y_test'], probs)
        assert auc > 0.5, f"Model AUC {auc:.3f} is not better than random"


# ─── Metrics Tests ────────────────────────────────────────────────────────────

class TestMetrics:
    
    def test_perfect_classifier_metrics(self):
        """Perfect classifier should return 1.0 for all metrics."""
        from src.models.train import compute_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['roc_auc'] == 1.0
    
    def test_metrics_all_keys_present(self):
        """Metrics dict should contain all expected keys."""
        from src.models.train import compute_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.9, 0.3])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        expected_keys = ['accuracy', 'precision', 'recall', 'f1', 
                         'roc_auc', 'specificity', 'npv', 'tp', 'tn', 'fp', 'fn']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_metrics_values_in_valid_range(self):
        """All rate metrics should be in [0, 1]."""
        from src.models.train import compute_metrics
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.randint(0, 2, 50)
        y_prob = np.random.random(50)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity']:
            assert 0.0 <= metrics[key] <= 1.0, f"Metric {key}={metrics[key]} out of range"


# ─── API Tests ────────────────────────────────────────────────────────────────

class TestPatientDataValidation:
    """Test Pydantic model validation without running the server."""
    
    def test_valid_patient_data(self):
        """Valid patient data should not raise errors."""
        from src.api.app import PatientData
        patient = PatientData(
            age=54, sex=1, cp=0, trestbps=130, chol=250,
            fbs=0, restecg=1, thalach=150, exang=0,
            oldpeak=1.5, slope=1, ca=0, thal=2
        )
        assert patient.age == 54
    
    def test_age_bounds_validation(self):
        """Age must be between 18 and 100."""
        from src.api.app import PatientData
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            PatientData(age=150, sex=1, cp=0, trestbps=130, chol=250,
                       fbs=0, restecg=1, thalach=150, exang=0,
                       oldpeak=1.5, slope=1, ca=0, thal=2)
    
    def test_sex_bounds_validation(self):
        """Sex must be 0 or 1."""
        from src.api.app import PatientData
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            PatientData(age=54, sex=5, cp=0, trestbps=130, chol=250,
                       fbs=0, restecg=1, thalach=150, exang=0,
                       oldpeak=1.5, slope=1, ca=0, thal=2)
    
    def test_feature_engineering_api(self):
        """API feature engineering should produce all expected fields."""
        from src.api.app import PatientData, engineer_patient_features
        patient = PatientData(
            age=54, sex=1, cp=0, trestbps=130, chol=250,
            fbs=0, restecg=1, thalach=150, exang=0,
            oldpeak=1.5, slope=1, ca=0, thal=2
        )
        features = engineer_patient_features(patient)
        
        expected_keys = ['hr_age_ratio', 'hr_reserve_pct', 'chol_age_interaction',
                         'hypertension_risk', 'st_depression_severe', 'multi_vessel',
                         'clinical_risk_score', 'age_group', 'cp_risk']
        for key in expected_keys:
            assert key in features, f"Missing engineered feature: {key}"
    
    def test_risk_factor_identification(self):
        """High-risk patients should have multiple identified factors."""
        from src.api.app import PatientData, get_risk_factors
        high_risk_patient = PatientData(
            age=65, sex=1, cp=0, trestbps=155, chol=310,
            fbs=1, restecg=2, thalach=120, exang=1,
            oldpeak=3.5, slope=2, ca=3, thal=3
        )
        factors = get_risk_factors(high_risk_patient)
        assert len(factors) >= 5, f"Expected ≥5 risk factors, got {len(factors)}"
    
    def test_recommendations_by_risk_level(self):
        """HIGH risk should produce more urgent recommendations."""
        from src.api.app import get_recommendations
        high_recs = get_recommendations('HIGH', [])
        low_recs = get_recommendations('LOW', [])
        
        high_text = ' '.join(high_recs).lower()
        assert 'immediate' in high_text or 'urgent' in high_text or '🚨' in high_text


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestEndToEnd:
    
    def test_full_mini_pipeline(self, sample_raw_data):
        """Run a small end-to-end pipeline without tuning."""
        # Preprocess
        cleaned = clean_data(sample_raw_data)
        engineered = engineer_features(cleaned)
        datasets = prepare_datasets(engineered, apply_smote=False)
        
        # Train minimal model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(datasets['X_train'], datasets['y_train'])
        
        probs = model.predict_proba(datasets['X_test'])[:, 1]
        preds = model.predict(datasets['X_test'])
        
        # Basic sanity checks
        assert len(preds) == len(datasets['y_test'])
        assert all(p in [0, 1] for p in preds)
        assert all(0 <= p <= 1 for p in probs)
        
        # AUC might be noisy with 10 samples, just check it's computed
        if len(np.unique(datasets['y_test'])) > 1:
            auc = roc_auc_score(datasets['y_test'], probs)
            assert 0 <= auc <= 1