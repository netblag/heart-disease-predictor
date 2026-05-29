"""
Heart Disease Prediction - Visualization Suite
==============================================
Comprehensive visualization of model performance, data exploration,
and clinical insights using Matplotlib and Plotly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import os

# Clinical color palette
PALETTE = {
    'red': '#E63946',
    'dark_blue': '#1D3557',
    'blue': '#457B9D',
    'light_blue': '#A8DADC',
    'cream': '#F1FAEE',
    'teal': '#2A9D8F',
    'orange': '#E76F51',
    'green': '#52B788',
    'purple': '#7B2D8B',
    'gold': '#F4A261'
}

MODEL_COLORS = [
    PALETTE['red'], PALETTE['blue'], PALETTE['teal'],
    PALETTE['orange'], PALETTE['purple'], PALETTE['gold'], PALETTE['green']
]


def plot_data_overview(df: pd.DataFrame, output_path: str):
    """
    Comprehensive EDA visualization with distribution plots and correlations.
    """
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(PALETTE['cream'])
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Title
    fig.suptitle('🫀 Heart Disease Dataset — Exploratory Analysis',
                 fontsize=20, fontweight='bold', color=PALETTE['dark_blue'], y=0.98)
    
    # 1. Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df['target'].value_counts()
    colors_pie = [PALETTE['teal'], PALETTE['red']]
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=['No Disease', 'Disease'],
        colors=colors_pie, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10}, pctdistance=0.85
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax1.set_title('Disease Distribution', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    
    # 2. Age distribution by disease
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE['cream'])
    for target_val, color, label in [(0, PALETTE['teal'], 'No Disease'), 
                                       (1, PALETTE['red'], 'Disease')]:
        subset = df[df['target'] == target_val]['age']
        ax2.hist(subset, bins=15, color=color, alpha=0.7, label=label, edgecolor='white')
    ax2.set_xlabel('Age', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Age Distribution by Diagnosis', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Cholesterol vs Max HR scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PALETTE['cream'])
    colors_scatter = df['target'].map({0: PALETTE['teal'], 1: PALETTE['red']})
    scatter = ax3.scatter(df['chol'], df['thalach'], c=colors_scatter, 
                          alpha=0.6, s=20, edgecolors='none')
    ax3.set_xlabel('Cholesterol (mg/dl)', fontsize=10)
    ax3.set_ylabel('Max Heart Rate', fontsize=10)
    ax3.set_title('Cholesterol vs Max Heart Rate', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    patch1 = mpatches.Patch(color=PALETTE['teal'], label='No Disease')
    patch2 = mpatches.Patch(color=PALETTE['red'], label='Disease')
    ax3.legend(handles=[patch1, patch2], fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Chest pain type
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PALETTE['cream'])
    cp_disease = df.groupby(['cp', 'target']).size().unstack(fill_value=0)
    cp_labels = ['Typical\nAngina', 'Atypical\nAngina', 'Non-Anginal\nPain', 'Asymptomatic']
    x = np.arange(len(cp_labels))
    w = 0.35
    ax4.bar(x - w/2, cp_disease.get(0, pd.Series(0, index=cp_disease.index)).values, 
            w, color=PALETTE['teal'], alpha=0.8, label='No Disease')
    ax4.bar(x + w/2, cp_disease.get(1, pd.Series(0, index=cp_disease.index)).values, 
            w, color=PALETTE['red'], alpha=0.8, label='Disease')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cp_labels, fontsize=8)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Chest Pain Type', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    ax4.legend(fontsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Feature correlation heatmap
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor(PALETTE['cream'])
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'clinical_risk_score', 'target']
    available_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()
    
    cmap = LinearSegmentedColormap.from_list('medical', 
        [PALETTE['teal'], 'white', PALETTE['red']], N=256)
    
    im = ax5.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax5.set_xticks(range(len(available_cols)))
    ax5.set_yticks(range(len(available_cols)))
    ax5.set_xticklabels(available_cols, rotation=45, ha='right', fontsize=9)
    ax5.set_yticklabels(available_cols, fontsize=9)
    
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            text = ax5.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black')
    
    ax5.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    plt.colorbar(im, ax=ax5, shrink=0.8)
    
    # 6. Blood pressure distribution
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor(PALETTE['cream'])
    for target_val, color, label in [(0, PALETTE['teal'], 'No Disease'), 
                                       (1, PALETTE['red'], 'Disease')]:
        subset = df[df['target'] == target_val]['trestbps']
        ax6.hist(subset, bins=20, color=color, alpha=0.7, label=label, edgecolor='white')
    ax6.axvline(x=140, color=PALETTE['dark_blue'], linestyle='--', linewidth=2, label='Hypertension Threshold')
    ax6.set_xlabel('Resting Blood Pressure', fontsize=10)
    ax6.set_title('Blood Pressure Distribution', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    ax6.legend(fontsize=8)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # 7. Clinical Risk Score distribution
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor(PALETTE['cream'])
    if 'clinical_risk_score' in df.columns:
        for target_val, color, label in [(0, PALETTE['teal'], 'No Disease'), 
                                           (1, PALETTE['red'], 'Disease')]:
            subset = df[df['target'] == target_val]['clinical_risk_score']
            ax7.hist(subset, bins=15, color=color, alpha=0.7, label=label, edgecolor='white')
        ax7.set_xlabel('Clinical Risk Score', fontsize=10)
        ax7.set_title('Clinical Risk Score Distribution', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
        ax7.legend(fontsize=9)
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
    
    # 8. Sex breakdown
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor(PALETTE['cream'])
    sex_disease = df.groupby(['sex', 'target']).size().unstack(fill_value=0)
    sex_labels = ['Female', 'Male']
    x = np.arange(2)
    ax8.bar(x - 0.2, sex_disease.get(0, [0,0]).values, 0.4, color=PALETTE['teal'], alpha=0.8, label='No Disease')
    ax8.bar(x + 0.2, sex_disease.get(1, [0,0]).values, 0.4, color=PALETTE['red'], alpha=0.8, label='Disease')
    ax8.set_xticks(x)
    ax8.set_xticklabels(sex_labels, fontsize=10)
    ax8.set_ylabel('Count', fontsize=10)
    ax8.set_title('Disease by Sex', fontsize=12, fontweight='bold', color=PALETTE['dark_blue'])
    ax8.legend(fontsize=9)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['cream'])
    plt.close()
    print(f"  💾 Saved: {output_path}")


def plot_roc_curves(results: dict, y_test, output_path: str):
    """
    Multi-model ROC curve comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(PALETTE['cream'])
    fig.suptitle('Model Performance Comparison', fontsize=18, 
                 fontweight='bold', color=PALETTE['dark_blue'], y=1.02)
    
    # ROC Curves
    ax = axes[0]
    ax.set_facecolor(PALETTE['cream'])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    for i, (name, res) in enumerate(results.items()):
        fpr = res['roc_curve']['fpr']
        tpr = res['roc_curve']['tpr']
        auc = res['test_metrics']['roc_auc']
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(fpr, tpr, linewidth=2.5, color=color, 
                label=f'{name} (AUC={auc:.3f})', alpha=0.85)
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold', color=PALETTE['dark_blue'])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Metrics bar chart
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE['cream'])
    
    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())
    x = np.arange(len(metrics_to_show))
    width = 0.8 / len(model_names)
    
    for i, (name, res) in enumerate(results.items()):
        m = res['test_metrics']
        values = [m[met] for met in metrics_to_show]
        offset = (i - len(model_names)/2) * width + width/2
        bars = ax2.bar(x + offset, values, width * 0.9, 
                       color=MODEL_COLORS[i % len(MODEL_COLORS)],
                       alpha=0.85, label=name)
    
    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold', color=PALETTE['dark_blue'])
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in metrics_to_show], fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.set_ylim([0, 1.15])
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['cream'])
    plt.close()
    print(f"  💾 Saved: {output_path}")


def plot_confusion_matrices(results: dict, y_test, output_path: str):
    """
    Confusion matrix grid for all models.
    """
    n_models = len(results)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.patch.set_facecolor(PALETTE['cream'])
    fig.suptitle('Confusion Matrices — All Models', fontsize=18, 
                 fontweight='bold', color=PALETTE['dark_blue'], y=1.01)
    
    axes = axes.flatten() if n_models > 1 else [axes]
    
    cmap = LinearSegmentedColormap.from_list('medical_cm', 
        ['white', PALETTE['dark_blue']], N=256)
    
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        ax.set_facecolor(PALETTE['cream'])
        
        y_pred = res['y_test_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        im = ax.imshow(cm, cmap=cmap, interpolation='nearest')
        
        thresh = cm.max() / 2.
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, format(cm[row, col], 'd'),
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       color='white' if cm[row, col] > thresh else PALETTE['dark_blue'])
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Disease', 'Disease'], fontsize=10)
        ax.set_yticklabels(['No Disease', 'Disease'], fontsize=10, rotation=90, va='center')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        auc = res['test_metrics']['roc_auc']
        ax.set_title(f'{name}\nAUC: {auc:.3f}', fontsize=11, fontweight='bold', color=PALETTE['dark_blue'])
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['cream'])
    plt.close()
    print(f"  💾 Saved: {output_path}")


def run_visualization_pipeline(df: pd.DataFrame, results: dict, 
                                 y_test, output_dir: str):
    """Run all visualization steps."""
    print("\n" + "=" * 60)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n  📈 EDA Overview...")
    plot_data_overview(df, os.path.join(output_dir, 'eda_overview.png'))
    
    print("  📈 ROC Curves & Metrics...")
    plot_roc_curves(results, y_test, os.path.join(output_dir, 'roc_comparison.png'))
    
    print("  📈 Confusion Matrices...")
    plot_confusion_matrices(results, y_test, os.path.join(output_dir, 'confusion_matrices.png'))
    
    print(f"\n  ✅ All plots saved to {output_dir}/")