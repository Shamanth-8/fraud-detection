"""
explainability.py
-----------------
SHAP-based explainability for GNN node features.
For each account, identifies the top-3 risk drivers.
Output: shap_explanations.csv
"""

import numpy as np
import pandas as pd
import shap
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "degree_centrality",
    "betweenness_centrality",
    "pagerank",
    "clustering_coeff",
    "eigenvector_centrality",
    "amount_zscore",
    "velocity",
    "entropy_risk",
    "cluster_density",
    "n_sent",
    "n_received",
    "total_sent",
    "mean_sent",
    "unique_receivers",
]

FEATURE_LABELS = {
    "degree_centrality": "high connectivity",
    "betweenness_centrality": "transfer hub",
    "pagerank": "network influence",
    "clustering_coeff": "fraud ring structure",
    "eigenvector_centrality": "high-influence account",
    "amount_zscore": "amount anomaly",
    "velocity": "velocity spike",
    "entropy_risk": "concentrated recipients",
    "cluster_density": "tight fraud cluster",
    "n_sent": "high send volume",
    "n_received": "high receive volume",
    "total_sent": "large total outflow",
    "mean_sent": "high avg transaction",
    "unique_receivers": "diverse receivers",
}

_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
OUTPUT_PATH = os.path.join(_ARTIFACTS, "shap_explanations.csv")


def build_feature_matrix(
    node_df: pd.DataFrame,
    quant_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = node_df.join(quant_df, how="left").fillna(0)
    available = [c for c in FEATURE_COLS if c in merged.columns]
    return merged[available]


def compute_shap_explanations(
    node_features_path: str = os.path.join(_ARTIFACTS, "node_features.csv"),
    quant_features_path: str = os.path.join(_ARTIFACTS, "quant_features.csv"),
    gnn_preds_path: str = os.path.join(_ARTIFACTS, "gnn_predictions.csv"),
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Use SHAP KernelExplainer on a surrogate Random Forest trained on node features.
    Outputs top-3 risk drivers per account.
    """
    print("Loading features …")
    node_df = pd.read_csv(node_features_path, index_col="account_id")
    quant_df = pd.read_csv(quant_features_path, index_col="account_id")

    X = build_feature_matrix(node_df, quant_df)
    feature_names = X.columns.tolist()

    # ── Train a surrogate model (RandomForest) for SHAP ──────────────────────
    if os.path.exists(gnn_preds_path):
        print("Using GNN predictions as labels for SHAP surrogate …")
        preds = pd.read_csv(gnn_preds_path, index_col="account_id")
        y = preds["fraud_predicted"].reindex(X.index).fillna(0).values
    else:
        print("Using fraud labels as fallback …")
        y = node_df["is_fraud"].reindex(X.index).fillna(0).values

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf.fit(X.values, y)
    print(f"  Surrogate RF trained (accuracy: {rf.score(X.values, y):.3f})")

    # ── SHAP TreeExplainer (fast for RF) ─────────────────────────────────────
    print("Computing SHAP values …")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X.values)

    # shap_values for class=1 (fraud) — handle multiple SHAP output formats
    if isinstance(shap_values, list):
        # Legacy format: list of 2 arrays, one per class
        sv = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # New format: shape (n_samples, n_features, n_classes)
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # ── Top-3 drivers per account ─────────────────────────────────────────────
    print("Extracting top-3 risk drivers …")
    records = []
    for i, acct in enumerate(X.index):
        shap_row = sv[i]
        top3_idx = np.argsort(np.abs(shap_row))[::-1][:3].tolist()
        top3_feats = [feature_names[j] for j in top3_idx]
        top3_vals = [float(shap_row[j]) for j in top3_idx]
        top3_labels = [FEATURE_LABELS.get(f, f) for f in top3_feats]

        records.append(
            {
                "account_id": acct,
                "driver_1": top3_labels[0],
                "driver_1_shap": round(top3_vals[0], 6),
                "driver_2": top3_labels[1] if len(top3_labels) > 1 else "",
                "driver_2_shap": round(top3_vals[1], 6) if len(top3_vals) > 1 else 0,
                "driver_3": top3_labels[2] if len(top3_labels) > 2 else "",
                "driver_3_shap": round(top3_vals[2], 6) if len(top3_vals) > 2 else 0,
                "summary": " | ".join(top3_labels[:3]),
            }
        )

    # Also save the full SHAP matrix for waterfall charts
    shap_df = pd.DataFrame(sv, index=X.index, columns=feature_names)
    shap_df.index.name = "account_id"
    shap_df.to_csv(os.path.join(_ARTIFACTS, "shap_values.csv"))

    result = pd.DataFrame(records).set_index("account_id")
    result.to_csv(output_path)

    print(f"\n✅ SHAP explanations saved → {output_path}")
    print(f"   SHAP values matrix  saved → shap_values.csv")
    print(f"\nSample explanations:")
    print(result.head(5)[["driver_1", "driver_2", "driver_3", "summary"]].to_string())
    return result, shap_df


if __name__ == "__main__":
    compute_shap_explanations()
