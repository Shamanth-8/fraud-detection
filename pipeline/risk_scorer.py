"""
risk_scorer.py
--------------
Computes the composite 0-100 Fraud Risk Score per account.

Formula:
  Risk Score = normalize(
    0.40 × GNN_fraud_probability
  + 0.20 × amount_zscore_risk
  + 0.15 × velocity_risk
  + 0.15 × network_centrality_risk
  + 0.10 × community_density_risk
  ) × 100
"""

import pandas as pd
import numpy as np
import os

_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1] using min-max scaling."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def compute_network_centrality_risk(node_df: pd.DataFrame) -> pd.Series:
    """
    Combine the 5 centrality metrics into a single centrality risk score.
    PageRank + betweenness + eigenvector are most indicative of hub behaviour.
    """
    centrality_cols = [
        "pagerank",
        "betweenness_centrality",
        "eigenvector_centrality",
        "degree_centrality",
    ]
    available = [c for c in centrality_cols if c in node_df.columns]
    if not available:
        return pd.Series(0.0, index=node_df.index)

    norm = pd.DataFrame({c: min_max_normalize(node_df[c]) for c in available})
    return norm.mean(axis=1)


def compute_risk_scores(
    node_df: pd.DataFrame,
    quant_df: pd.DataFrame,
    gnn_probs: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute composite risk scores.

    Parameters
    ----------
    node_df   : DataFrame with graph + centrality features (index = account_id)
    quant_df  : DataFrame with quant features  (index = account_id)
    gnn_probs : Series of GNN fraud probabilities (index = account_id), optional

    Returns
    -------
    DataFrame with risk_score (0-100) and component breakdown columns.
    """
    idx = node_df.index

    # ── Component 1: GNN probability ─────────────────────────────────────────
    if gnn_probs is not None and len(gnn_probs) > 0:
        gnn_risk = min_max_normalize(gnn_probs.reindex(idx).fillna(0))
    else:
        # Fallback: use fraud label if available (for demo without trained GNN)
        if "is_fraud" in node_df.columns:
            gnn_risk = node_df["is_fraud"].astype(float)
        else:
            gnn_risk = pd.Series(0.0, index=idx)

    # ── Component 2: Amount z-score risk ─────────────────────────────────────
    amount_risk = min_max_normalize(quant_df["amount_zscore"].reindex(idx).fillna(0))

    # ── Component 3: Velocity risk ───────────────────────────────────────────
    velocity_risk = min_max_normalize(quant_df["velocity"].reindex(idx).fillna(0))

    # ── Component 4: Network centrality risk ─────────────────────────────────
    centrality_risk = compute_network_centrality_risk(node_df)

    # ── Component 5: Cluster density risk ────────────────────────────────────
    density_risk = min_max_normalize(quant_df["cluster_density"].reindex(idx).fillna(0))

    # ── Composite score ───────────────────────────────────────────────────────
    raw_score = (
        0.40 * gnn_risk
        + 0.20 * amount_risk
        + 0.15 * velocity_risk
        + 0.15 * centrality_risk
        + 0.10 * density_risk
    )

    risk_score = (min_max_normalize(raw_score) * 100).round(2)

    result = pd.DataFrame(
        {
            "risk_score": risk_score,
            "gnn_probability": (gnn_risk * 100).round(2),
            "amount_risk": (amount_risk * 100).round(2),
            "velocity_risk": (velocity_risk * 100).round(2),
            "centrality_risk": (centrality_risk * 100).round(2),
            "community_density_risk": (density_risk * 100).round(2),
            "is_fraud": node_df.get("is_fraud", pd.Series(0, index=idx))
            .reindex(idx)
            .fillna(0),
            "community_id": node_df.get("community_id", pd.Series(-1, index=idx))
            .reindex(idx)
            .fillna(-1),
        },
        index=idx,
    )

    return result


def load_and_score(
    node_features_path: str = os.path.join(_ARTIFACTS, "node_features.csv"),
    quant_features_path: str = os.path.join(_ARTIFACTS, "quant_features.csv"),
    gnn_probs_path: str | None = os.path.join(_ARTIFACTS, "gnn_predictions.csv"),
    output_path: str = os.path.join(_ARTIFACTS, "risk_scores.csv"),
) -> pd.DataFrame:
    print("Loading node features …")
    node_df = pd.read_csv(node_features_path, index_col="account_id")

    print("Loading quant features …")
    quant_df = pd.read_csv(quant_features_path, index_col="account_id")

    gnn_probs = None
    if gnn_probs_path and os.path.exists(gnn_probs_path):
        print("Loading GNN predictions …")
        gnn_preds = pd.read_csv(gnn_probs_path, index_col="account_id")
        gnn_probs = gnn_preds["fraud_probability"]
    else:
        print("GNN predictions not found — using fraud label as proxy.")

    print("Computing risk scores …")
    scores = compute_risk_scores(node_df, quant_df, gnn_probs)
    scores.to_csv(output_path)

    print(f"\n✅ Risk scores saved → {output_path}")
    print(
        f"   Score range: {scores['risk_score'].min():.1f} – {scores['risk_score'].max():.1f}"
    )
    print(f"\nTop 10 riskiest accounts:")
    top10 = scores.nlargest(10, "risk_score")[
        [
            "risk_score",
            "gnn_probability",
            "amount_risk",
            "velocity_risk",
            "centrality_risk",
            "community_density_risk",
            "is_fraud",
        ]
    ]
    print(top10.to_string())
    return scores


if __name__ == "__main__":
    load_and_score()
