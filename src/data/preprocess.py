"""
Heart Disease Dataset Preprocessing Pipeline
============================================
Handles data loading, cleaning, feature engineering, and preparation
for model training using the UCI Heart Disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# UCI Heart Disease dataset column names
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1=male, 0=female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1=true)',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1=yes)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Number of major vessels (0-3)',
    'thal': 'Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)',
    'target': 'Heart disease presence (1=disease, 0=no disease)'
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data with proper column names."""
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    print(f"✅ Loaded {len(df)} records with {df.shape[1]} features")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset: handle missing values, fix data types, remove outliers.
    """
    df = df.copy()
    
    # Replace '?' with NaN (UCI dataset uses '?' for missing)
    df.replace('?', np.nan, inplace=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\n🔍 Missing values before cleaning:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Impute missing values
    for col in ['ca', 'thal']:
        df[col] = df[col].fillna(df[col].median())
    
    # Binarize target: 0 = no disease, 1 = disease (values > 0)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Remove clear outliers (domain knowledge)
    df = df[df['trestbps'] > 0]   # Blood pressure can't be 0
    df = df[df['chol'] > 0]       # Cholesterol can't be 0
    df = df[df['thalach'] > 0]    # Heart rate can't be 0
    
    print(f"\n✅ Cleaned dataset: {len(df)} records remaining")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed engineered features for improved model performance.
    
    Medical domain knowledge applied:
    - Pulse pressure: indicator of arterial stiffness
    - Age-HR interaction: max HR naturally declines with age
    - Cholesterol-age ratio: risk compound effect
    - Clinical risk score: composite of multiple factors
    """
    df = df.copy()
    
    # --- Cardiovascular Risk Features ---
    
    # Pulse pressure indicator (proxy using available features)
    df['hr_age_ratio'] = df['thalach'] / df['age']
    
    # Expected max HR based on age (220 - age formula)
    df['expected_max_hr'] = 220 - df['age']
    df['hr_reserve_pct'] = (df['thalach'] / df['expected_max_hr']) * 100
    
    # Cholesterol risk category
    df['chol_age_interaction'] = df['chol'] * df['age'] / 1000
    
    # High blood pressure indicator
    df['hypertension_risk'] = ((df['trestbps'] >= 140) | 
                                (df['trestbps'] >= 130)).astype(int)
    
    # ST depression severity
    df['st_depression_severe'] = (df['oldpeak'] >= 2.0).astype(int)
    
    # Multi-vessel disease indicator
    df['multi_vessel'] = (df['ca'] >= 2).astype(int)
    
    # Composite clinical risk score
    df['clinical_risk_score'] = (
        (df['age'] > 55).astype(int) * 2 +
        df['sex'] * 1.5 +
        (df['cp'] == 0).astype(int) * 2 +
        (df['trestbps'] > 140).astype(int) * 1 +
        (df['chol'] > 240).astype(int) * 1 +
        df['exang'].astype(int) * 2 +
        (df['oldpeak'] > 1).astype(int) * 1.5 +
        df['ca'] * 1.5
    )
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 40, 50, 60, 70, 100],
                              labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Chest pain severity (0=asymptomatic is highest risk)
    df['cp_risk'] = df['cp'].map({0: 3, 1: 2, 2: 1, 3: 0}).fillna(0).astype(int)
    
    print(f"✅ Engineered {df.shape[1] - 14} new features")
    return df


def prepare_datasets(df: pd.DataFrame, 
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     random_state: int = 42,
                     apply_smote: bool = True):
    """
    Split data into train/validation/test sets with optional SMOTE oversampling.
    
    Returns:
        dict with X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols]
    y = df['target']
    
    print(f"\n📊 Class distribution:")
    print(y.value_counts())
    print(f"   Imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, 
        random_state=random_state, stratify=y_trainval
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE on training data only
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"\n✅ After SMOTE - Training samples: {len(y_train)}")
    
    print(f"\n📋 Dataset splits:")
    print(f"   Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler,
        'X_train_df': X_train,  # unscaled for SHAP
        'X_test_df': X_test,
    }


def run_pipeline(input_path: str, output_dir: str) -> dict:
    """Full preprocessing pipeline."""
    print("=" * 60)
    print("🫀 HEART DISEASE PREDICTOR - Data Pipeline")
    print("=" * 60)
    
    df_raw = load_raw_data(input_path)
    df_clean = clean_data(df_raw)
    df_engineered = engineer_features(df_clean)
    
    # Save processed data
    import os
    os.makedirs(output_dir, exist_ok=True)
    df_engineered.to_csv(f"{output_dir}/processed_data.csv", index=False)
    print(f"\n💾 Saved processed data to {output_dir}/processed_data.csv")
    
    datasets = prepare_datasets(df_engineered)
    return datasets, df_engineered