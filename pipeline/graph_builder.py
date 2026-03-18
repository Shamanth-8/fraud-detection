"""
graph_builder.py
----------------
Builds a directed NetworkX transaction graph.
Computes node-level graph features (5 centrality metrics + Louvain communities).
Output: node_features.csv
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain
import pickle
import os

_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
INPUT_FILE = os.path.join(_ARTIFACTS, "graph_transactions.csv")
OUTPUT_FILE = os.path.join(_ARTIFACTS, "node_features.csv")
GRAPH_FILE = os.path.join(_ARTIFACTS, "transaction_graph.gpickle")


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed weighted graph from the transaction DataFrame."""
    G = nx.DiGraph()

    # Add all accounts as nodes (with fraud label)
    account_fraud = {}
    for _, row in df.iterrows():
        s, r = row["sender_id"], row["receiver_id"]
        if s not in account_fraud:
            account_fraud[s] = 0
        if r not in account_fraud:
            account_fraud[r] = 0
        if row["is_fraud"] == 1:
            account_fraud[s] = 1
            account_fraud[r] = 1

    for acct, label in account_fraud.items():
        G.add_node(acct, is_fraud=label)

    # Add edges (aggregate: total amount, count)
    edge_data = (
        df.groupby(["sender_id", "receiver_id"])
        .agg(
            total_amount=("amount", "sum"),
            txn_count=("amount", "count"),
            is_fraud=("is_fraud", "max"),
        )
        .reset_index()
    )

    for _, row in edge_data.iterrows():
        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            weight=row["total_amount"],
            count=row["txn_count"],
            is_fraud=row["is_fraud"],
        )

    return G


def compute_centrality(G: nx.DiGraph) -> dict:
    """Compute 5 centrality metrics for each node."""
    print("  Computing degree centrality …")
    deg = nx.degree_centrality(G)

    print("  Computing betweenness centrality …")
    between = nx.betweenness_centrality(G, normalized=True, weight="weight")

    print("  Computing PageRank …")
    pagerank = nx.pagerank(G, weight="weight", max_iter=200)

    print("  Computing clustering coefficient (undirected) …")
    G_und = G.to_undirected()
    cluster = nx.clustering(G_und, weight="weight")

    print("  Computing eigenvector centrality …")
    try:
        eigenv = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eigenv = {n: 0.0 for n in G.nodes()}

    return {
        "degree_centrality": deg,
        "betweenness_centrality": between,
        "pagerank": pagerank,
        "clustering_coeff": cluster,
        "eigenvector_centrality": eigenv,
    }


def run_louvain(G: nx.DiGraph) -> dict:
    """Run Louvain community detection on undirected graph."""
    print("  Running Louvain community detection …")
    G_und = G.to_undirected()
    # Weight by txn_count
    for u, v, data in G_und.edges(data=True):
        if "count" not in data:
            data["count"] = 1
    partition = community_louvain.best_partition(G_und, weight="count", random_state=42)
    return partition


def compute_node_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account transaction stats (used as additional GNN features)."""
    sender_stats = (
        df.groupby("sender_id")
        .agg(
            total_sent=("amount", "sum"),
            n_sent=("amount", "count"),
            mean_sent=("amount", "mean"),
            std_sent=("amount", "std"),
            unique_receivers=("receiver_id", "nunique"),
            fraud_sent=("is_fraud", "sum"),
        )
        .rename_axis("account_id")
    )

    recv_stats = (
        df.groupby("receiver_id")
        .agg(
            total_received=("amount", "sum"),
            n_received=("amount", "count"),
            unique_senders=("sender_id", "nunique"),
        )
        .rename_axis("account_id")
    )

    stats = sender_stats.join(recv_stats, how="outer").fillna(0)
    return stats


def main():
    print(f"Loading {INPUT_FILE} …")
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    print(
        f"  → {len(df):,} transactions, {df['sender_id'].nunique() + df['receiver_id'].nunique()} unique account refs"
    )

    # Ground-truth fraud label per account
    fraud_accounts = set(
        df[df["is_fraud"] == 1]["sender_id"].tolist()
        + df[df["is_fraud"] == 1]["receiver_id"].tolist()
    )

    # Build graph
    print("Building NetworkX graph …")
    G = build_graph(df)
    print(f"  → Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    # Centrality metrics
    print("Computing centrality metrics …")
    centrality = compute_centrality(G)

    # Louvain
    partition = run_louvain(G)
    n_communities = len(set(partition.values()))
    print(f"  → {n_communities} communities detected")

    # Node transaction stats
    print("Computing per-account transaction stats …")
    stats = compute_node_stats(df)

    # ── Assemble node feature DataFrame ──────────────────────────────────────
    nodes = list(G.nodes())
    records = []
    for node in nodes:
        row = {
            "account_id": node,
            "is_fraud": 1 if node in fraud_accounts else 0,
            "community_id": partition.get(node, -1),
            "degree_centrality": centrality["degree_centrality"].get(node, 0),
            "betweenness_centrality": centrality["betweenness_centrality"].get(node, 0),
            "pagerank": centrality["pagerank"].get(node, 0),
            "clustering_coeff": centrality["clustering_coeff"].get(node, 0),
            "eigenvector_centrality": centrality["eigenvector_centrality"].get(node, 0),
        }
        if node in stats.index:
            s = stats.loc[node]
            row.update(
                {
                    "total_sent": s.get("total_sent", 0),
                    "n_sent": s.get("n_sent", 0),
                    "mean_sent": s.get("mean_sent", 0),
                    "std_sent": s.get("std_sent", 0),
                    "unique_receivers": s.get("unique_receivers", 0),
                    "total_received": s.get("total_received", 0),
                    "n_received": s.get("n_received", 0),
                    "unique_senders": s.get("unique_senders", 0),
                }
            )
        else:
            row.update(
                {
                    "total_sent": 0,
                    "n_sent": 0,
                    "mean_sent": 0,
                    "std_sent": 0,
                    "unique_receivers": 0,
                    "total_received": 0,
                    "n_received": 0,
                    "unique_senders": 0,
                }
            )
        records.append(row)

    node_df = pd.DataFrame(records).set_index("account_id")
    node_df.to_csv(OUTPUT_FILE)

    # Save graph for later use in dashboard
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n✅ Node features saved → {OUTPUT_FILE}")
    print(f"   Graph saved         → {GRAPH_FILE}")
    print(f"\nFeature summary:")
    print(node_df.describe().round(4).to_string())
    print(f"\nFraud accounts in graph: {node_df['is_fraud'].sum()} / {len(node_df)}")
    print(f"Communities: {node_df['community_id'].nunique()}")


if __name__ == "__main__":
    main()
