"""
quant_features_runner.py
------------------------
Standalone runner: loads node_features.csv + graph_transactions.csv,
computes all quant features, and saves quant_features.csv.
"""

import os
import sys
import pandas as pd

# Ensure project root is on sys.path for 'pipeline.quant_features' import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.quant_features import compute_all_quant_features

_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
NODE_PATH = os.path.join(_ARTIFACTS, "node_features.csv")
TXN_PATH = os.path.join(_ARTIFACTS, "graph_transactions.csv")
OUT_PATH = os.path.join(_ARTIFACTS, "quant_features.csv")

if __name__ == "__main__":
    print("Loading data …")
    node_df = pd.read_csv(NODE_PATH, index_col="account_id")
    df_txn = pd.read_csv(TXN_PATH, parse_dates=["timestamp"])

    print("Computing quantitative features …")
    quant_df = compute_all_quant_features(df_txn, node_df)

    quant_df.to_csv(OUT_PATH)
    print(f"\n✅ Quant features saved → {OUT_PATH}")
    print(f"   Shape: {quant_df.shape}")
    print(f"\nFeature summary:")
    print(quant_df.describe().round(4).to_string())
