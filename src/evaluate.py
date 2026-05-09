from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


# ── Colour palette (consistent across plots) ─────────────────────────────────
PALETTE = {
    'primary':   '#4C6EF5',
    'secondary': '#F03E3E',
    'accent':    '#37B24D',
    'bg':        '#F8F9FA',
    'text':      '#212529',
}

os.makedirs('outputs/plots', exist_ok=True)


# ── 1. Per-model metrics ─────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test,
                   model_name: str = "Model") -> dict:

    y_pred = model.predict(X_test)

    # Probability scores for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    metrics = {
        'Model':     model_name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, y_score),
    }
    return metrics, y_pred, y_score


# ── 2. Confusion Matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred,
                          model_name: str = "Model",
                          save: bool = True) -> str:
    """
    Plot and optionally save a confusion matrix.

    Returns the file path if saved, else empty string.
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Non-Suicide', 'Suicide']

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor='white',
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual', fontsize=12, labelpad=10)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, pad=15)
    fig.tight_layout()

    path = ''
    if save:
        safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        path = f'outputs/plots/cm_{safe}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 3. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc_curves(models_dict: dict,
                    X_test, y_test,
                    save: bool = True) -> str:
    """
    Plot ROC curves for all models on a single axes.

    Parameters
    ----------
    models_dict : {model_name: fitted_estimator}

    Returns path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models_dict)))

    for (name, model), color in zip(models_dict.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict(X_test).astype(float)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, label=f'{name}  (AUC={auc:.3f})', color=color, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = ''
    if save:
        path = 'outputs/plots/roc_curves.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 4. Model Comparison Table ─────────────────────────────────────────────────

def compare_models(models_dict: dict,
                   X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a sorted comparison DataFrame.
    """
    rows = []
    for name, model in models_dict.items():
        metrics, _, _ = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index('Model')
    df = df.sort_values('F1-Score', ascending=False)
    return df


def plot_model_comparison(comparison_df: pd.DataFrame,
                          save: bool = True) -> str:
    """
    Bar chart comparing all models across key metrics.
    """
    metrics_to_plot = ['Accuracy', 'Precision',
                       'Recall', 'F1-Score', 'ROC-AUC']
    df = comparison_df[metrics_to_plot].reset_index()
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric',
                palette='tab10', ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — All Metrics', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=9)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = ''
    if save:
        path = 'outputs/plots/model_comparison.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 5. Full evaluation report ─────────────────────────────────────────────────

def full_evaluation(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """
    Run complete evaluation: metrics table + confusion matrices + ROC curves.

    Returns the comparison DataFrame.
    """
    print("\n" + "═"*60)
    print("  MODEL EVALUATION REPORT")
    print("═"*60)

    for name, model in models_dict.items():
        metrics, y_pred, _ = evaluate_model(model, X_test, y_test, name)
        print(f"\n▸ {name}")
        print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  F1-Score  : {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}")
        print("\n" + classification_report(
            y_test, y_pred,
            target_names=['Non-Suicide', 'Suicide']
        ))
        plot_confusion_matrix(y_test, y_pred, model_name=name)

    print("\n" + "─"*60)
    comparison = compare_models(models_dict, X_test, y_test)
    print("\nComparison Table (sorted by F1-Score):")
    print(comparison.to_string(float_format="{:.4f}".format))

    roc_path = plot_roc_curves(models_dict, X_test, y_test)
    comp_path = plot_model_comparison(comparison)
    print(f"\nPlots saved → {roc_path}, {comp_path}")

    return comparison


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=300, n_features=20, random_state=42)
    X_tr, X_te = X[:240], X[240:]
    y_tr, y_te = y[:240], y[240:]

    model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    comp = full_evaluation({'Logistic Regression': model}, X_te, y_te)
    print(comp)
