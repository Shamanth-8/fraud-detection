#!/usr/bin/env python3
"""
evaluate_model.py
-----------------
Standalone evaluation script for GraphSAGE Fraud Detection.
Evaluates the PyTorch Geometric GraphSAGE model against a Baseline Random Forest,
computing AUC-PR, AUC-ROC, F1-scores, Precision, Recall, Confusion Matrices,
and generating visualization plots (w/ SHAP Feature Importances).

Usage:
    python evaluate_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Import GNN components to re-use same logic and ensure test split matches
from pipeline.gnn_model import build_pyg_data, build_model, _import_torch, FEATURE_COLS

def main():
    torch, nn, F = _import_torch()
    
    # 1. Ensure output directories exist
    os.makedirs("results/plots", exist_ok=True)
    
    # 2. Load data
    print("Loading datasets...")
    node_features_path = "artifacts/node_features.csv"
    quant_features_path = "artifacts/quant_features.csv"
    txns_path = "artifacts/graph_transactions.csv"
    
    if not os.path.exists(node_features_path):
        print("Required artifacts missing. Please run the pipeline first.")
        return
        
    node_df = pd.read_csv(node_features_path, index_col="account_id")
    quant_df = pd.read_csv(quant_features_path, index_col="account_id")
    df_txn = pd.read_csv(txns_path, parse_dates=["timestamp"])
    
    # Recreate the data object and test split exactly as in training
    print("Building Graph object and ensuring identical train/test split...")
    data, all_accounts, n_features = build_pyg_data(node_df, quant_df, df_txn)
    
    labels = data.y.numpy()
    n = data.num_nodes
    indices = np.arange(n)
    
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Extract features and labels required
    X_all = data.x.numpy()
    y_test = labels[test_idx]
    
    # Rebuild dynamic feature names for plotting
    merged = node_df.join(quant_df, how="left").fillna(0)
    available_cols = [c for c in FEATURE_COLS if c in merged.columns]
    
    # 3. Load GraphSAGE Model and Run Inference
    print("Loading trained GraphSAGE model...")
    gnn_model = build_model(n_features)
    
    model_path = "artifacts/gnn_fraud_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run the pipeline first.")
        
    gnn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gnn_model.eval()
    
    print("Evaluating GraphSAGE on test set...")
    with torch.no_grad():
        out = gnn_model(data.x, data.edge_index)
        gnn_probs_all = torch.softmax(out, dim=1)[:, 1].numpy()
        
    gnn_probs = gnn_probs_all[test_idx]
    gnn_preds = (gnn_probs > 0.5).astype(int)
    
    # 4. Train Baseline (Random Forest)
    print("Training Baseline Model (Random Forest)...")
    X_train_base = X_all[train_idx]
    y_train_base = labels[train_idx]
    X_test_base = X_all[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train_base, y_train_base)
    
    rf_probs = rf.predict_proba(X_test_base)[:, 1]
    rf_preds = rf.predict(X_test_base)
    
    # 5. Compute Metrics
    print("Computing metrics...")
    
    def compute_all_metrics(y_true, y_pred, y_prob):
        return {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "precision_class_0": float(precision_score(y_true, y_pred, pos_label=0)),
            "recall_class_0": float(recall_score(y_true, y_pred, pos_label=0)),
            "precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "auc_pr": float(average_precision_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
    gnn_metrics = compute_all_metrics(y_test, gnn_preds, gnn_probs)
    rf_metrics = compute_all_metrics(y_test, rf_preds, rf_probs)
    
    results = {
        "GraphSAGE": gnn_metrics,
        "Baseline_RF": rf_metrics
    }
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Results saved to results/evaluation_results.json")
    
    # 6. Generate Plots
    print("Generating plots...")
    
    # A) PR Curve
    plt.figure(figsize=(8, 6))
    precision_gnn, recall_gnn, _ = precision_recall_curve(y_test, gnn_probs)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)
    
    plt.plot(recall_gnn, precision_gnn, label=f"GraphSAGE (AUC-PR = {gnn_metrics['auc_pr']:.3f})", color="royalblue", lw=2)
    plt.plot(recall_rf, precision_rf, label=f"Baseline RF (AUC-PR = {rf_metrics['auc_pr']:.3f})", linestyle="--", color="crimson", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Fraud Class)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/plots/pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # B) ROC Curve
    plt.figure(figsize=(8, 6))
    fpr_gnn, tpr_gnn, _ = roc_curve(y_test, gnn_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    
    plt.plot(fpr_gnn, tpr_gnn, label=f"GraphSAGE (AUC-ROC = {gnn_metrics['auc_roc']:.3f})", color="royalblue", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Baseline RF (AUC-ROC = {rf_metrics['auc_roc']:.3f})", linestyle="--", color="crimson", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/plots/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # C) Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(gnn_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("GraphSAGE Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    axes[0].set_yticklabels(["Non-Fraud (0)", "Fraud (1)"])
    
    sns.heatmap(rf_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title("Baseline RF Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    axes[1].set_yticklabels(["Non-Fraud (0)", "Fraud (1)"])
    
    plt.tight_layout()
    plt.savefig("results/plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # D) SHAP Feature Importance Bar Chart (Baseline Model)
    print("Computing SHAP values for top 10 features...")
    # Use TreeExplainer for the Random Forest
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_base)
    
    # Retrieve SHAP values for class 1 (Fraud) safely based on formatting
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv_class1 = shap_values[:, :, 1]
    else:
        sv_class1 = shap_values
        
    if hasattr(sv_class1, "values"):
        sv_class1 = sv_class1.values
        
    # Calculate mean absolute shap values for bar chart
    mean_abs_shap = np.mean(np.abs(sv_class1), axis=0)
    shap_df = pd.DataFrame({
        "Feature": available_cols,
        "Importance": mean_abs_shap
    }).sort_values(by="Importance", ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=shap_df, hue="Feature", palette="viridis", legend=False)
    plt.title("Top 10 Feature Importances (SHAP values - Baseline RF)")
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model Output)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("results/plots/shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Plots saved in results/plots/")
    
    # 7. Print Summary to Console
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Metric':<22} | {'GraphSAGE':<14} | {'Baseline RF':<14}")
    print("-" * 60)
    
    metrics_to_print = [
        ("AUC-PR (Focus)", "auc_pr"),
        ("AUC-ROC", "auc_roc"),
        ("F1 Macro", "f1_macro"),
        ("F1 Weighted", "f1_weighted"),
        ("Precision (Fraud)", "precision_class_1"),
        ("Recall (Fraud)", "recall_class_1"),
        ("Precision (Non-Fraud)", "precision_class_0"),
        ("Recall (Non-Fraud)", "recall_class_0"),
    ]
    
    for name, key in metrics_to_print:
        print(f"{name:<22} | {gnn_metrics[key]:<14.4f} | {rf_metrics[key]:<14.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
