"""
quant_features.py
-----------------
Proper quantitative risk analytics for fraud detection.
Computes statistical/financial risk metrics per account.
"""

import pandas as pd
import numpy as np


def compute_amount_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Per-account average z-score of transaction amounts.
    Accounts sending/receiving unusually large amounts get high scores.
    z = (amount - global_mean) / global_std
    """
    global_mean = df["amount"].mean()
    global_std = df["amount"].std()
    df = df.copy()
    df["amount_zscore"] = (df["amount"] - global_mean) / global_std

    # Sender-side: mean absolute z-score across all sent transactions
    sender_z = (
        df.groupby("sender_id")["amount_zscore"]
        .apply(lambda x: x.abs().mean())
        .rename("amount_zscore_sender")
    )

    receiver_z = (
        df.groupby("receiver_id")["amount_zscore"]
        .apply(lambda x: x.abs().mean())
        .rename("amount_zscore_receiver")
    )

    combined = pd.concat([sender_z, receiver_z], axis=1)
    combined.index.name = "account_id"
    combined["amount_zscore"] = combined[
        ["amount_zscore_sender", "amount_zscore_receiver"]
    ].max(axis=1)
    return combined["amount_zscore"].fillna(0)


def compute_transaction_velocity(df: pd.DataFrame, window_hours: int = 1) -> pd.Series:
    """
    Transaction velocity: peak transaction rate relative to hourly average.
    velocity = peak_txns_in_window / average_txns_per_window
    Higher velocity → burst activity → suspicious.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_bucket"] = df["timestamp"].dt.floor(f"{window_hours}h")

    # Count per sender per hour bucket
    hourly_counts = (
        df.groupby(["sender_id", "hour_bucket"]).size().reset_index(name="count")
    )

    # Peak and mean counts per account
    velocity_stats = hourly_counts.groupby("sender_id")["count"].agg(
        peak_count="max", mean_count="mean"
    )
    velocity_stats["velocity"] = velocity_stats["peak_count"] / (
        velocity_stats["mean_count"] + 1e-6
    )
    velocity_stats.index.name = "account_id"
    return velocity_stats["velocity"].fillna(0)


def compute_entropy(df: pd.DataFrame) -> pd.Series:
    """
    Shannon entropy of transaction destinations per sender.
    Low entropy → sender concentrates transactions to few accounts → suspicious.
    H = -sum(p_i * log(p_i))
    We return (1 - normalized_entropy) so that LOW diversity = HIGH risk.
    """

    def shannon_entropy(series):
        counts = series.value_counts()
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-12))

    entropy = df.groupby("sender_id")["receiver_id"].apply(shannon_entropy)
    # Normalize: max possible entropy = log(n_unique_receivers)
    max_receivers = df.groupby("sender_id")["receiver_id"].nunique()
    max_entropy = np.log(max_receivers + 1e-12)
    normalized = entropy / (max_entropy + 1e-12)
    # Invert: low diversity → high risk
    risk = 1 - normalized.clip(0, 1)
    risk.index.name = "account_id"
    return risk.fillna(0.5)


def compute_cluster_density(
    node_df: pd.DataFrame, df_transactions: pd.DataFrame
) -> pd.Series:
    """
    Community/cluster internal density per account.
    density = internal_edges / (n_nodes * (n_nodes - 1))
    Tight clusters = potential fraud rings.
    """
    if "community_id" not in node_df.columns:
        return pd.Series(0.0, index=node_df.index, name="cluster_density")

    # Map account → community
    acct_community = node_df["community_id"].to_dict()
    df2 = df_transactions.copy()
    df2["sender_comm"] = df2["sender_id"].map(acct_community)
    df2["receiver_comm"] = df2["receiver_id"].map(acct_community)

    # Internal edges (same community)
    internal = df2[df2["sender_comm"] == df2["receiver_comm"]]
    comm_internal_count = (
        internal.groupby("sender_comm").size().rename("internal_edges")
    )

    # Community sizes
    comm_sizes = node_df.groupby("community_id").size().rename("comm_size")

    comm_stats = pd.concat([comm_sizes, comm_internal_count], axis=1).fillna(0)
    comm_stats["max_edges"] = comm_stats["comm_size"] * (comm_stats["comm_size"] - 1)
    comm_stats["density"] = (
        comm_stats["internal_edges"] / (comm_stats["max_edges"] + 1e-6)
    ).clip(0, 1)

    # Assign community density back to each account
    density_map = comm_stats["density"].to_dict()
    result = node_df["community_id"].map(density_map).fillna(0)
    result.name = "cluster_density"
    return result


def compute_all_quant_features(
    df_transactions: pd.DataFrame, node_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute all 4 quantitative features and merge into node_df index.
    Returns a DataFrame with columns: amount_zscore, velocity, entropy_risk, cluster_density
    """
    print("  [Quant] Computing amount z-scores …")
    amount_z = compute_amount_zscore(df_transactions)

    print("  [Quant] Computing transaction velocity …")
    velocity = compute_transaction_velocity(df_transactions)

    print("  [Quant] Computing entropy risk …")
    entropy = compute_entropy(df_transactions)

    print("  [Quant] Computing cluster density …")
    density = compute_cluster_density(node_df, df_transactions)

    quant = pd.DataFrame(
        {
            "amount_zscore": amount_z,
            "velocity": velocity,
            "entropy_risk": entropy,
            "cluster_density": density,
        }
    )
    quant.index.name = "account_id"

    # Align with node_df index
    quant = quant.reindex(node_df.index).fillna(0)
    return quant
