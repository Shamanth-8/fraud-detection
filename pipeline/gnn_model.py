"""
gnn_model.py
------------
GraphSAGE-based GNN for node-level fraud classification.
Uses node features from graph_builder.py + quant_features.py.
Outputs: gnn_fraud_model.pt, gnn_predictions.csv
"""

import os
import numpy as np
import pandas as pd


# ── Lazy imports so the file can be imported without torch ───────────────────
def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _import_pyg():
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import from_networkx

    return torch_geometric, Data, SAGEConv, from_networkx


# ── Feature columns used as GNN input ────────────────────────────────────────
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

_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
MODEL_PATH = os.path.join(_ARTIFACTS, "gnn_fraud_model.pt")
PREDS_PATH = os.path.join(_ARTIFACTS, "gnn_predictions.csv")


class GraphSAGEFraud(object.__class__):
    pass


def build_pyg_data(node_df: pd.DataFrame, quant_df: pd.DataFrame, df_txn: pd.DataFrame):
    """Build a PyTorch Geometric Data object from node features and edge list."""
    torch, nn, F = _import_torch()
    _, Data, _, _ = _import_pyg()

    # Align indices
    all_accounts = node_df.index.tolist()
    acct2idx = {a: i for i, a in enumerate(all_accounts)}

    # Node feature matrix
    merged = node_df.join(quant_df, how="left").fillna(0)
    available_cols = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available_cols].values.astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float)

    # Labels
    y = node_df["is_fraud"].values.astype(np.int64)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Edge index
    edges_src, edges_dst = [], []
    for _, row in df_txn.iterrows():
        s = acct2idx.get(row["sender_id"])
        r = acct2idx.get(row["receiver_id"])
        if s is not None and r is not None:
            edges_src.append(s)
            edges_dst.append(r)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.num_classes = 2
    return data, all_accounts, len(available_cols)


def build_model(in_channels: int):
    """Build GraphSAGE model."""
    torch, nn, F = _import_torch()
    _, _, SAGEConv, _ = _import_pyg()

    class _SAGEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, 64)
            self.conv2 = SAGEConv(64, 32)
            self.conv3 = SAGEConv(32, 16)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(32)
            self.lin = nn.Linear(16, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            return self.lin(x)

    return _SAGEModel()


def train_gnn(
    node_features_path: str = os.path.join(_ARTIFACTS, "node_features.csv"),
    quant_features_path: str = os.path.join(_ARTIFACTS, "quant_features.csv"),
    txn_path: str = os.path.join(_ARTIFACTS, "graph_transactions.csv"),
    epochs: int = 80,
    lr: float = 0.005,
):
    torch, nn, F = _import_torch()
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    from sklearn.model_selection import train_test_split

    print("Loading features …")
    node_df = pd.read_csv(node_features_path, index_col="account_id")
    quant_df = pd.read_csv(quant_features_path, index_col="account_id")
    df_txn = pd.read_csv(txn_path, parse_dates=["timestamp"])

    print("Building PyG data …")
    data, all_accounts, n_features = build_pyg_data(node_df, quant_df, df_txn)
    print(
        f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, Features: {n_features}"
    )

    n = data.num_nodes
    indices = np.arange(n)
    labels = data.y.numpy()

    # Stratified split (handle imbalance)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Class weights for imbalance
    n_fraud = labels.sum()
    n_legit = n - n_fraud
    weight = torch.tensor([1.0, n_legit / (n_fraud + 1e-6)], dtype=torch.float)

    model = build_model(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=weight)

    print(f"Training GraphSAGE for {epochs} epochs …")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(out, dim=1)[:, 1].numpy()
                preds = (probs > 0.5).astype(int)
                auc = roc_auc_score(labels[test_idx], probs[test_idx])
                f1 = f1_score(labels[test_idx], preds[test_idx], zero_division=0)
            print(
                f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test AUC: {auc:.4f} | F1: {f1:.4f}"
            )
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(labels[test_idx], probs[test_idx])
    f1 = f1_score(labels[test_idx], preds[test_idx], zero_division=0)
    print(f"\n✅ Final Test AUC: {auc:.4f}  |  F1: {f1:.4f}")
    print(
        classification_report(
            labels[test_idx], preds[test_idx], target_names=["Normal", "Fraud"]
        )
    )

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "account_id": all_accounts,
            "fraud_probability": probs,
            "fraud_predicted": preds,
        }
    ).set_index("account_id")
    pred_df.to_csv(PREDS_PATH)
    print(f"Predictions saved → {PREDS_PATH}")

    return model, pred_df


if __name__ == "__main__":
    train_gnn()
