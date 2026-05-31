"""
Heart Disease Prediction - Model Explainability with SHAP
==========================================================
Generates SHAP-based feature importance, individual predictions explanations,
and global model interpretability visualizations.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import os

# Medical color palette
COLORS = {
    'primary': '#E63946',
    'secondary': '#1D3557',
    'accent': '#457B9D',
    'background': '#F1FAEE',
    'positive': '#2A9D8F',
    'negative': '#E76F51',
    'neutral': '#264653'
}


def compute_shap_values(model, X_data, feature_names: list, model_name: str = ''):
    """
    Compute SHAP values using the appropriate explainer for each model type.
    
    Returns shap_values array and explainer object.
    """
    print(f"  🔬 Computing SHAP values for {model_name}...", end='', flush=True)
    
    X_df = pd.DataFrame(X_data, columns=feature_names) if not isinstance(X_data, pd.DataFrame) else X_data
    
    model_type = type(model).__name__.lower()
    
    try:
        if any(name in model_type for name in ['xgb', 'lgbm', 'lightgbm', 'xgboost']):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif 'randomforest' in model_type or 'gradientboosting' in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Use KernelExplainer for SVM, Logistic Regression, etc.
            background = shap.kmeans(X_df, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_df[:100], nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
    
    except Exception as e:
        print(f"\n  ⚠️  SHAP computation failed: {e}")
        return None, None
    
    print(" Done!")
    return shap_values, explainer


def plot_global_importance(shap_values, feature_names: list, 
                            output_path: str, top_n: int = 15) -> pd.DataFrame:
    """
    Plot global feature importance based on mean absolute SHAP values.
    """
    # Ensure shap_values is 2D
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[1]  # Take class 1 for multi-output
    
    # Compute mean absolute SHAP (flatten if needed)
    mean_shap = np.abs(sv).mean(axis=0).flatten()
    
    n_features = len(feature_names)
    mean_shap = mean_shap[:n_features]
    
    importance_df = pd.DataFrame({
        'feature': list(feature_names[:len(mean_shap)]),
        'importance': mean_shap.tolist()
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Color bars by importance level
    colors = [COLORS['primary'] if i < 5 else COLORS['accent'] 
              if i < 10 else COLORS['neutral'] 
              for i in range(len(importance_df))]
    
    bars = ax.barh(range(len(importance_df)), importance_df['importance'].values,
                   color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values, fontsize=11)
    ax.invert_yaxis()
    
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12, color=COLORS['neutral'])
    ax.set_title('🫀 Global Feature Importance (SHAP)', fontsize=16, 
                 fontweight='bold', color=COLORS['secondary'], pad=15)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_df['importance'].values)):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9, 
                color=COLORS['neutral'])
    
    # Legend
    patches = [
        mpatches.Patch(color=COLORS['primary'], label='Top 5 Features'),
        mpatches.Patch(color=COLORS['accent'], label='Features 6-10'),
        mpatches.Patch(color=COLORS['neutral'], label='Features 11-15'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return importance_df


def plot_shap_summary(shap_values, X_data, feature_names: list, output_path: str):
    """
    SHAP beeswarm summary plot showing feature impact direction.
    """
    X_df = pd.DataFrame(X_data, columns=feature_names) if not isinstance(X_data, pd.DataFrame) else X_data
    
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[1]
    
    # Align dimensions
    n_samples = min(sv.shape[0], len(X_df))
    n_feats = min(sv.shape[1] if sv.ndim > 1 else sv.shape[0], len(feature_names))
    sv = sv[:n_samples, :n_feats]
    X_plot = X_df.iloc[:n_samples, :n_feats].copy()
    X_plot.columns = feature_names[:n_feats]

    shap.summary_plot(
        sv, X_plot,
        plot_type='dot',
        max_display=15,
        show=False,
        color_bar=True,
        plot_size=(12, 9)
    )
    
    plt.title('SHAP Summary Plot - Feature Impact on Heart Disease Prediction',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {output_path}")


def explain_single_prediction(model, patient_data: dict, feature_names: list,
                               scaler, shap_values_row=None) -> dict:
    """
    Generate human-readable explanation for a single patient prediction.
    """
    # Prepare input
    X = pd.DataFrame([patient_data])[feature_names].values
    X_scaled = scaler.transform(X)
    
    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(prob >= 0.5)
    
    risk_level = 'HIGH' if prob >= 0.7 else 'MODERATE' if prob >= 0.4 else 'LOW'
    risk_color = '🔴' if risk_level == 'HIGH' else '🟡' if risk_level == 'MODERATE' else '🟢'
    
    explanation = {
        'prediction': pred,
        'probability': float(prob),
        'risk_level': risk_level,
        'risk_icon': risk_color,
        'patient_data': patient_data,
    }
    
    return explanation


def run_explainability_pipeline(trained_models: dict, datasets: dict, 
                                  best_model_name: str, output_dir: str):
    """
    Run full explainability analysis on the best model.
    """
    print("\n" + "=" * 60)
    print("🔬 MODEL EXPLAINABILITY ANALYSIS (SHAP)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    best_model = trained_models[best_model_name]
    X_test = datasets['X_test']
    feature_names = datasets['feature_names']
    
    # Use test data as DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Compute SHAP
    shap_values, explainer = compute_shap_values(
        best_model, X_test_df, feature_names, best_model_name
    )
    
    if shap_values is None:
        print("  ⚠️  Skipping SHAP plots (computation failed)")
        return None
    
    # Global importance plot
    importance_df = plot_global_importance(
        shap_values, feature_names,
        os.path.join(output_dir, 'shap_feature_importance.png')
    )
    
    # Summary plot
    plot_shap_summary(
        shap_values, X_test_df, feature_names,
        os.path.join(output_dir, 'shap_summary.png')
    )
    
    print(f"\n  🏆 Top 5 Most Important Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"     {i+1}. {row['feature']:<30} SHAP: {row['importance']:.4f}")
    
    return shap_values, importance_df