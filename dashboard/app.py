"""
app.py
------
Graph-Based Financial Fraud Detection & Quantitative Risk Analytics Platform
6-page Streamlit dashboard:
  1. Fraud Overview
  2. Transaction Network (PyVis full graph + subgraph drill-down)
  3. Fraud Rings (Louvain communities)
  4. Temporal Analysis
  5. Quant Analytics
  6. Account Inspector (SHAP + subgraph)
"""

import os
import pickle
import hashlib
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network
import tempfile

warnings.filterwarnings("ignore")

# ── Artifacts directory ──────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "artifacts"
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FraudNet Analytics",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f5;
}
[data-testid="stSidebar"] .stRadio label { color: #ffffff !important; font-size: 0.95rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #c4b5fd !important; }
[data-testid="stSidebar"] > div > div > div > .stMarkdown p,
[data-testid="stSidebar"] > div > div > div > .stMarkdown span,
[data-testid="stSidebar"] > div > div > div > .stMarkdown label,
[data-testid="stSidebar"] > div > div > div > .stMarkdown small { color: #e8e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown { color: #f0f0f5 !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #c4b5fd !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ffffff !important; }
/* File uploader text stays dark on its white background */
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section small { color: #6b7280 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button { color: #4f46e5 !important; }
/* Upload label stays light */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label { color: #e8e8f0 !important; }

/* Main bg */
.main { background: #0d1117; color: #e6edf3; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1f2e, #252b3b);
    border-radius: 12px;
    border: 1px solid #30363d;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"]  { color: #a78bfa !important; font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricDelta"]  { color: #34d399 !important; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, #6d28d9, #2563eb);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(109,40,217,0.3);
}
.page-header h1 { color: white !important; margin: 0; font-size: 1.7rem; font-weight: 700; }
.page-header p  { color: rgba(255,255,255,0.75) !important; margin: 4px 0 0; }

/* Risk badge */
.badge-critical { background:#ef4444; color:white; border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-high     { background:#f97316; color:white; border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-medium   { background:#eab308; color:#111; border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.badge-low      { background:#22c55e; color:white; border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }

/* Driver cards */
.driver-card {
    background: linear-gradient(135deg, #1e2130, #252b3b);
    border-radius: 10px;
    border-left: 4px solid #a78bfa;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  IN-APP PIPELINE (runs on uploaded CSV)
# ══════════════════════════════════════════════════════════════════════════════

import sys

# Ensure project root is on path so pipeline modules are importable
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.graph_builder import (
    build_graph,
    compute_centrality,
    run_louvain,
    compute_node_stats,
)
from pipeline.quant_features import compute_all_quant_features
from pipeline.risk_scorer import compute_risk_scores


# ── SHAP in-memory (lightweight, no file I/O) ───────────────────────────────

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


def _compute_shap_in_memory(node_df, quant_df):
    """Compute SHAP explanations in-memory using surrogate Random Forest."""
    import shap
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    merged = node_df.join(quant_df, how="left").fillna(0)
    available = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available]
    feature_names = X.columns.tolist()

    y = node_df["is_fraud"].reindex(X.index).fillna(0).values

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf.fit(X.values, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X.values)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

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

    shap_df = pd.DataFrame(sv, index=X.index, columns=feature_names)
    shap_df.index.name = "account_id"

    result = pd.DataFrame(records).set_index("account_id")
    return result, shap_df


def convert_kaggle_credit_card_format(df):
    """
    Detect and convert Kaggle-style credit card fraud datasets.
    These have columns: Time, V1-V28, Amount, Class.
    We synthesise transaction_id, sender_id, receiver_id, amount, timestamp, is_fraud.
    """
    cols = set(df.columns)
    has_class = "Class" in cols
    has_v_features = all(f"V{i}" in cols for i in range(1, 29))
    has_amount = "Amount" in cols
    has_time = "Time" in cols

    # Also handle lowercase variants
    if not has_class:
        has_class = "class" in cols
    if not has_amount:
        has_amount = "amount" in cols

    if not (has_class and has_v_features):
        return df, False  # Not a Kaggle CC dataset

    st.info("🔄 Detected **Kaggle Credit Card Fraud** format — auto-converting to graph transaction format …")

    out = pd.DataFrame()
    n = len(df)

    # transaction_id
    out["transaction_id"] = [f"TXN{i:06d}" for i in range(n)]

    # Synthesise sender_id & receiver_id from V-features
    # Use a hash of the first few V-features to create plausible account clusters
    n_accounts = max(200, n // 50)  # ~50 txns per account on average

    def _hash_to_account(row, prefix, features):
        raw = "_".join(f"{row[f]:.4f}" for f in features)
        h = int(hashlib.md5(raw.encode()).hexdigest(), 16)
        return f"{prefix}{h % n_accounts:05d}"

    sender_features = ["V1", "V2", "V3", "V4", "V5"]
    receiver_features = ["V6", "V7", "V8", "V9", "V10"]

    out["sender_id"] = df.apply(lambda r: _hash_to_account(r, "ACC_S", sender_features), axis=1)
    out["receiver_id"] = df.apply(lambda r: _hash_to_account(r, "ACC_R", receiver_features), axis=1)

    # Amount
    amount_col = "Amount" if "Amount" in df.columns else "amount"
    out["amount"] = df[amount_col].astype(float)

    # Timestamp — convert 'Time' (seconds from first txn) into real datetimes
    time_col = "Time" if "Time" in df.columns else "time"
    if time_col in df.columns:
        base_time = pd.Timestamp("2024-01-01")
        out["timestamp"] = base_time + pd.to_timedelta(df[time_col].astype(float), unit="s")
    else:
        # Fallback: generate sequential timestamps
        out["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="30s")

    # Fraud label
    class_col = "Class" if "Class" in df.columns else "class"
    out["is_fraud"] = df[class_col].astype(int)

    fraud_count = out["is_fraud"].sum()
    st.success(
        f"✅ Converted **{n:,}** transactions  •  "
        f"**{out['sender_id'].nunique():,}** senders  •  "
        f"**{out['receiver_id'].nunique():,}** receivers  •  "
        f"**{fraud_count:,}** fraud labels ({fraud_count/n*100:.1f}%)"
    )

    return out, True


def run_pipeline_on_upload(txn_df):
    """
    Run the full analysis pipeline in-memory on an uploaded DataFrame.
    Returns dict with all analysis results + the NetworkX graph.
    """
    progress = st.progress(0, text="Building transaction graph …")

    # Step 1: Build graph
    G = build_graph(txn_df)
    progress.progress(15, text="Computing centrality metrics …")

    # Step 2: Centrality
    centrality = compute_centrality(G)
    progress.progress(40, text="Running community detection …")

    # Step 3: Louvain communities
    partition = run_louvain(G)
    progress.progress(50, text="Computing per-account stats …")

    # Step 4: Node stats
    stats = compute_node_stats(txn_df)

    # Assemble node features
    fraud_accounts = set(
        txn_df[txn_df["is_fraud"] == 1]["sender_id"].tolist()
        + txn_df[txn_df["is_fraud"] == 1]["receiver_id"].tolist()
    )
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
    progress.progress(60, text="Computing quantitative features …")

    # Step 5: Quant features
    quant_df = compute_all_quant_features(txn_df, node_df)
    progress.progress(70, text="Computing risk scores …")

    # Step 6: Risk scores (no GNN — uses is_fraud label as proxy)
    scores_df = compute_risk_scores(node_df, quant_df, gnn_probs=None)
    progress.progress(80, text="Computing SHAP explainability …")

    # Step 7: SHAP
    try:
        shap_expl, shap_vals = _compute_shap_in_memory(node_df, quant_df)
    except Exception:
        shap_expl, shap_vals = None, None

    progress.progress(100, text="✅ Pipeline complete!")

    return {
        "txn": txn_df,
        "nodes": node_df,
        "quant": quant_df,
        "scores": scores_df,
        "shap": shap_expl,
        "shap_v": shap_vals,
        "gnn": None,
        "graph": G,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (cached — for demo/pre-computed data)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def load_data():
    data = {}
    files = {
        "txn": "graph_transactions.csv",
        "nodes": "node_features.csv",
        "quant": "quant_features.csv",
        "scores": "risk_scores.csv",
        "shap": "shap_explanations.csv",
        "shap_v": "shap_values.csv",
        "gnn": "gnn_predictions.csv",
    }
    for key, fname in files.items():
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(fpath):
            idx_col = "account_id" if key != "txn" else None
            data[key] = pd.read_csv(fpath, index_col=idx_col)
        else:
            data[key] = None

    # Parse timestamp
    if data["txn"] is not None:
        data["txn"]["timestamp"] = pd.to_datetime(data["txn"]["timestamp"])

    return data


@st.cache_resource(show_spinner=False)
def load_graph():
    gpath = os.path.join(ARTIFACTS_DIR, "transaction_graph.gpickle")
    if os.path.exists(gpath):
        with open(gpath, "rb") as f:
            return pickle.load(f)
    return None


def check_data_ready():
    return all(
        [
            os.path.exists(os.path.join(ARTIFACTS_DIR, "graph_transactions.csv")),
            os.path.exists(os.path.join(ARTIFACTS_DIR, "node_features.csv")),
            os.path.exists(os.path.join(ARTIFACTS_DIR, "risk_scores.csv")),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def risk_color(score: float) -> str:
    if score >= 75:
        return "#ef4444"
    if score >= 50:
        return "#f97316"
    if score >= 25:
        return "#eab308"
    return "#22c55e"


def risk_label(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def build_pyvis_graph(
    G: nx.DiGraph,
    scores_df: pd.DataFrame,
    max_nodes: int = 400,
    highlight_accounts: list = None,
    subgraph_center: str = None,
    subgraph_hops: int = 1,
) -> str:
    """
    Render a PyVis network graph and return the HTML string.
    If subgraph_center is set, show only that account's ego network.
    """
    if subgraph_center and subgraph_center in G:
        # Ego subgraph
        nodes_in = set()
        nodes_in.add(subgraph_center)
        for hop in range(subgraph_hops):
            new_nodes = set()
            for n in list(nodes_in):
                new_nodes.update(G.predecessors(n))
                new_nodes.update(G.successors(n))
            nodes_in.update(new_nodes)
        G_view = G.subgraph(nodes_in)
    else:
        # Sample large graphs
        all_nodes = list(G.nodes())
        if len(all_nodes) > max_nodes:
            # Prioritize high-risk accounts
            if scores_df is not None:
                top_risk = scores_df.nlargest(
                    max_nodes // 2, "risk_score"
                ).index.tolist()
                rest = [n for n in all_nodes if n not in top_risk]
                import random

                selected = top_risk + random.sample(
                    rest, min(max_nodes // 2, len(rest))
                )
            else:
                selected = all_nodes[:max_nodes]
            G_view = G.subgraph(selected)
        else:
            G_view = G

    net = Network(
        height="580px",
        width="100%",
        bgcolor="#0d1117",
        font_color="#c9d1d9",
        directed=True,
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 120},
        "minVelocity": 0.75
      },
      "edges": {
        "color": {"color": "#30363d", "highlight": "#a78bfa"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "continuous"}
      },
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    for node in G_view.nodes():
        score = 0.0
        is_fr = 0
        comm = -1
        if scores_df is not None and node in scores_df.index:
            score = float(scores_df.loc[node, "risk_score"])
            is_fr = int(scores_df.loc[node, "is_fraud"])
            comm = int(scores_df.loc[node, "community_id"])

        color = risk_color(score)
        size = 10 + score / 8

        # Highlight
        if highlight_accounts and node in highlight_accounts:
            border_color = "#f0f6ff"
            border_width = 3
        elif node == subgraph_center:
            border_color = "#ffffff"
            border_width = 4
            size = 25
        else:
            border_color = color
            border_width = 1

        tooltip = (
            f"<b>{node}</b><br>"
            f"Risk Score: <b>{score:.1f}</b><br>"
            f"Label: {'🚨 FRAUD' if is_fr else '✅ Normal'}<br>"
            f"Community: {comm}"
        )

        net.add_node(
            node,
            label=node if subgraph_center else "",
            title=tooltip,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": "#a78bfa"},
            },
            size=size,
            borderWidth=border_width,
        )

    for u, v, edata in G_view.edges(data=True):
        w = float(edata.get("weight", 1))
        net.add_edge(u, v, value=min(w / 1000, 5), title=f"${w:,.0f}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        return open(f.name).read()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — dual mode: Upload CSV or use demo data
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🕵️ FraudNet Analytics")
    st.markdown("---")

    # ── Data Source Selector ─────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        help="CSV must have: transaction_id, sender_id, receiver_id, amount, timestamp, is_fraud",
    )

    with st.expander("📋 Required CSV Format", expanded=False):
        st.markdown("""
**Your CSV must have these 6 columns:**

| Column | Example |
|---|---|
| `transaction_id` | `TXN001` |
| `sender_id` | `ACC_123` |
| `receiver_id` | `ACC_456` |
| `amount` | `1500.00` |
| `timestamp` | `2024-01-15 14:30:00` |
| `is_fraud` | `0` or `1` |

✅ **Works with:** Transaction / payment / wire fraud datasets

✅ **Kaggle Credit Card datasets** (Time, V1-V28, Amount, Class) are **auto-converted** — just upload directly!
""")

    demo_available = check_data_ready()
    if demo_available:
        use_demo = st.button("📦 Use Demo Dataset", use_container_width=True)
    else:
        use_demo = False

    st.markdown("---")

    # ── Handle upload ────────────────────────────────────────────────────────
    data_loaded = False

    if uploaded_file is not None:
        # Check if we already processed this file
        file_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_pipeline_key") != file_key:
            # New file — parse and validate
            try:
                raw_df = pd.read_csv(uploaded_file)

                # Auto-detect and convert Kaggle credit card format
                raw_df, was_converted = convert_kaggle_credit_card_format(raw_df)

                required = {
                    "transaction_id",
                    "sender_id",
                    "receiver_id",
                    "amount",
                    "timestamp",
                    "is_fraud",
                }
                missing = required - set(raw_df.columns)
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                    st.info(
                        "Required: transaction_id, sender_id, receiver_id, amount, timestamp, is_fraud"
                    )
                    st.stop()

                raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
                st.info(
                    f"📄 **{uploaded_file.name}** — {len(raw_df):,} transactions, "
                    f"{raw_df['sender_id'].nunique():,} senders"
                )

                # Run pipeline
                results = run_pipeline_on_upload(raw_df)

                # Store in session state
                st.session_state["pipeline_data"] = results
                st.session_state["pipeline_graph"] = results.pop("graph")
                st.session_state["_pipeline_key"] = file_key
                st.session_state["_data_source"] = "upload"
                st.rerun()

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        # Already processed — load from session state
        data = st.session_state["pipeline_data"]
        G = st.session_state["pipeline_graph"]
        data_loaded = True

    elif use_demo and demo_available:
        # Clear any uploaded data and switch to demo
        st.session_state.pop("pipeline_data", None)
        st.session_state.pop("pipeline_graph", None)
        st.session_state.pop("_pipeline_key", None)
        st.session_state["_data_source"] = "demo"
        st.rerun()

    if not data_loaded:
        # Try session state (from previous upload)
        if "pipeline_data" in st.session_state:
            data = st.session_state["pipeline_data"]
            G = st.session_state["pipeline_graph"]
            data_loaded = True
        elif demo_available:
            data = load_data()
            G = load_graph()
            data_loaded = True

    if not data_loaded:
        st.warning("⚠️ Upload a transaction CSV to get started.")
        st.markdown("**Required CSV columns:**")
        st.code("transaction_id, sender_id, receiver_id, amount, timestamp, is_fraud")
        st.stop()

    scores_df = data["scores"]

    # Show summary
    source = st.session_state.get("_data_source", "demo")
    if source == "upload":
        st.success("✅ Uploaded data analysed")
    else:
        st.success("✅ Demo data loaded")

    if scores_df is not None:
        n_critical = (scores_df["risk_score"] >= 75).sum()
        n_high = (
            (scores_df["risk_score"] >= 50) & (scores_df["risk_score"] < 75)
        ).sum()
        st.metric("🚨 Critical Risk", n_critical)
        st.metric("⚠️ High Risk", n_high)

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "📊 Fraud Overview",
            "🌐 Transaction Network",
            "🔴 Fraud Rings",
            "⏱️ Temporal Analysis",
            "📐 Quant Analytics",
            "🔍 Account Inspector",
        ],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — FRAUD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Fraud Overview":
    st.markdown(
        """
    <div class="page-header">
        <h1>📊 Fraud Overview Dashboard</h1>
        <p>Platform-wide fraud risk metrics, score distribution, and top risky accounts</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    txn_df = data["txn"]
    nodes_df = data["nodes"]
    scores_df = data["scores"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Accounts", f"{len(scores_df):,}")
    c2.metric("Total Transactions", f"{len(txn_df):,}")
    c3.metric("Fraud Accounts", f"{int(scores_df['is_fraud'].sum()):,}")
    c4.metric("Critical Risk (≥75)", f"{int((scores_df['risk_score'] >= 75).sum()):,}")
    c5.metric("Avg Risk Score", f"{scores_df['risk_score'].mean():.1f}")

    st.markdown("---")

    col1, col2 = st.columns([1.6, 1])

    # Risk score distribution
    with col1:
        fig = px.histogram(
            scores_df,
            x="risk_score",
            nbins=50,
            color_discrete_sequence=["#7c3aed"],
            title="Risk Score Distribution",
            labels={"risk_score": "Risk Score (0-100)", "count": "Accounts"},
        )
        fig.add_vline(
            x=50,
            line_dash="dash",
            line_color="#f97316",
            annotation_text="High Risk Threshold",
            annotation_position="top right",
        )
        fig.add_vline(
            x=75,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Critical Threshold",
            annotation_position="top right",
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font_color="#c9d1d9",
            title_font_size=16,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fraud vs normal pie
    with col2:
        labels_pie = ["Normal Accounts", "Fraud Accounts"]
        values_pie = [
            int((scores_df["is_fraud"] == 0).sum()),
            int(scores_df["is_fraud"].sum()),
        ]
        fig_pie = go.Figure(
            go.Pie(
                labels=labels_pie,
                values=values_pie,
                hole=0.55,
                marker=dict(colors=["#22c55e", "#ef4444"]),
                textinfo="label+percent",
            )
        )
        fig_pie.update_layout(
            title="Account Classification",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font_color="#c9d1d9",
            title_font_size=16,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Top risky accounts
    st.markdown("### 🚨 Top 20 Riskiest Accounts")
    top20 = scores_df.nlargest(20, "risk_score").reset_index()
    top20["Risk Level"] = top20["risk_score"].apply(risk_label)

    fig_bar = px.bar(
        top20,
        x="account_id",
        y="risk_score",
        color="risk_score",
        color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        range_color=[0, 100],
        hover_data=[
            "gnn_probability",
            "amount_risk",
            "velocity_risk",
            "centrality_risk",
        ],
        title="Top 20 Riskiest Accounts",
        labels={"risk_score": "Risk Score", "account_id": "Account"},
    )
    fig_bar.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font_color="#c9d1d9",
        xaxis_tickangle=45,
        coloraxis_showscale=True,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📋 Full Top-20 Table"):
        display_cols = [
            "account_id",
            "risk_score",
            "gnn_probability",
            "amount_risk",
            "velocity_risk",
            "centrality_risk",
            "community_density_risk",
            "is_fraud",
            "Risk Level",
        ]
        st.dataframe(
            top20[
                [c for c in display_cols if c in top20.columns]
            ].style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TRANSACTION NETWORK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌐 Transaction Network":
    st.markdown(
        """
    <div class="page-header">
        <h1>🌐 Transaction Network Graph</h1>
        <p>Interactive graph — node colour = risk score. Hover for details.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if G is None:
        st.error("Graph file not found. Run `graph_builder.py` first.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        max_nodes = st.slider("Max nodes to render", 100, 600, 350, 50)
    with col2:
        filter_high_risk = st.checkbox(
            "Show only high-risk accounts (≥50)", value=False
        )

    scores_df = data["scores"]
    render_df = scores_df
    if filter_high_risk and scores_df is not None:
        high_risk = scores_df[scores_df["risk_score"] >= 50].index.tolist()
        render_df = scores_df.loc[high_risk]

    with st.spinner("Rendering network graph …"):
        html_str = build_pyvis_graph(G, render_df, max_nodes=max_nodes)
    st.components.v1.html(html_str, height=600, scrolling=False)

    st.markdown("---")
    st.markdown("### 🔎 Subgraph Drill-Down")
    st.caption("Select an account to explore its immediate transaction neighborhood.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        account_input = st.text_input("Enter Account ID (e.g. ACC00042)", "")
    with col_b:
        hops = st.selectbox("Neighbourhood hops", [1, 2], index=0)

    if account_input:
        if account_input in G:
            with st.spinner(f"Building subgraph for {account_input} …"):
                subgraph_html = build_pyvis_graph(
                    G,
                    scores_df,
                    subgraph_center=account_input,
                    subgraph_hops=hops,
                )
            st.info(f"Showing {hops}-hop neighbourhood of **{account_input}**")
            st.components.v1.html(subgraph_html, height=500, scrolling=False)

            # Show account stats
            if scores_df is not None and account_input in scores_df.index:
                row = scores_df.loc[account_input]
                c1, c2, c3 = st.columns(3)
                c1.metric("Risk Score", f"{row['risk_score']:.1f}")
                c2.metric("Fraud Label", "🚨 Fraud" if row["is_fraud"] else "✅ Normal")
                c3.metric("Community", int(row["community_id"]))
        else:
            st.warning(f"Account `{account_input}` not found in the graph.")

    # Legend
    st.markdown("---")
    st.markdown("**Colour Legend:**")
    cols = st.columns(4)
    for c, label, colour in zip(
        cols,
        [
            "🟢 Low (0-25)",
            "🟡 Medium (25-50)",
            "🟠 High (50-75)",
            "🔴 Critical (75-100)",
        ],
        ["#22c55e", "#eab308", "#f97316", "#ef4444"],
    ):
        c.markdown(
            f"<span style='color:{colour}; font-weight:600'>{label}</span>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — FRAUD RINGS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔴 Fraud Rings":
    st.markdown(
        """
    <div class="page-header">
        <h1>🔴 Fraud Ring Detection</h1>
        <p>Louvain community detection reveals clusters of collaborating accounts</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    scores_df = data["scores"]
    nodes_df = data["nodes"]

    if scores_df is None:
        st.error("Run `risk_scorer.py` first.")
        st.stop()

    # Community stats
    comm_stats = (
        scores_df.groupby("community_id")
        .agg(
            n_accounts=("risk_score", "count"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            fraud_accounts=("is_fraud", "sum"),
            avg_gnn_risk=("gnn_probability", "mean"),
            avg_density=("community_density_risk", "mean"),
        )
        .reset_index()
    )
    comm_stats["fraud_rate"] = (
        comm_stats["fraud_accounts"] / comm_stats["n_accounts"] * 100
    ).round(1)
    comm_stats = comm_stats.sort_values("avg_risk_score", ascending=False)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Communities", len(comm_stats))
    c2.metric(
        "Suspicious Communities (fraud>0)",
        int((comm_stats["fraud_accounts"] > 0).sum()),
    )
    c3.metric("Largest Community", int(comm_stats["n_accounts"].max()))

    st.markdown("---")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        # Scatter: size = accounts, color = avg risk
        fig = px.scatter(
            comm_stats,
            x="n_accounts",
            y="avg_risk_score",
            size="n_accounts",
            color="avg_risk_score",
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
            hover_data=["community_id", "fraud_accounts", "fraud_rate"],
            title="Community Risk vs. Size",
            labels={"n_accounts": "Community Size", "avg_risk_score": "Avg Risk Score"},
        )
        fig.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font_color="#c9d1d9",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar: top 15 communities by fraud rate
        top15_comm = comm_stats.nlargest(15, "fraud_rate")
        fig2 = px.bar(
            top15_comm,
            x="community_id",
            y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale=["#22c55e", "#ef4444"],
            title="Top 15 Communities by Fraud Rate (%)",
            labels={"fraud_rate": "Fraud Rate (%)", "community_id": "Community ID"},
        )
        fig2.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font_color="#c9d1d9",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Density vs fraud probability
    st.markdown("### 📈 Cluster Density vs. Fraud Probability")
    fig3 = px.scatter(
        scores_df.reset_index(),
        x="community_density_risk",
        y="gnn_probability",
        color="risk_score",
        color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
        opacity=0.6,
        hover_data=["account_id", "is_fraud", "community_id"],
        title="Cluster Density vs GNN Fraud Probability",
        labels={
            "community_density_risk": "Cluster Density Risk",
            "gnn_probability": "GNN Risk (%)",
        },
    )
    fig3.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font_color="#c9d1d9",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📋 Suspicious Community Summary")
    suspicious = comm_stats[comm_stats["fraud_accounts"] > 0].copy()
    suspicious["avg_risk_score"] = suspicious["avg_risk_score"].round(1)
    st.dataframe(
        suspicious.style.background_gradient(
            subset=["avg_risk_score", "fraud_rate"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⏱️ Temporal Analysis":
    st.markdown(
        """
    <div class="page-header">
        <h1>⏱️ Temporal Analysis</h1>
        <p>Transaction patterns over time — fraud bursts, suspicious activity windows</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    txn_df = data["txn"]
    if txn_df is None:
        st.error("Run `graph_data_generator.py` first.")
        st.stop()

    txn_df["date"] = txn_df["timestamp"].dt.date
    txn_df["hour"] = txn_df["timestamp"].dt.hour
    txn_df["dow"] = txn_df["timestamp"].dt.day_name()
    txn_df["week"] = txn_df["timestamp"].dt.isocalendar().week.astype(int)

    # ── Daily transaction volume ───────────────────────────────────────────────
    daily = txn_df.groupby(["date", "is_fraud"]).size().reset_index(name="count")
    daily["label"] = daily["is_fraud"].map({0: "Normal", 1: "Fraud"})

    fig = px.line(
        daily,
        x="date",
        y="count",
        color="label",
        color_discrete_map={"Normal": "#22c55e", "Fraud": "#ef4444"},
        title="Daily Transaction Volume",
        labels={"date": "Date", "count": "Transactions", "label": "Type"},
    )
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font_color="#c9d1d9",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # Hourly heatmap (hour × day-of-week)
    with col1:
        heat_data = (
            txn_df[txn_df["is_fraud"] == 1]
            .groupby(["dow", "hour"])
            .size()
            .reset_index(name="count")
        )
        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        heat_data["dow"] = pd.Categorical(
            heat_data["dow"], categories=order, ordered=True
        )
        heat_pivot = heat_data.pivot(
            index="dow", columns="hour", values="count"
        ).fillna(0)
        fig2 = px.imshow(
            heat_pivot,
            color_continuous_scale="Reds",
            title="Fraud Transaction Burst Heatmap (Hour × Day of Week)",
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Fraud Txns"},
            aspect="auto",
        )
        fig2.update_layout(paper_bgcolor="#161b22", font_color="#c9d1d9")
        st.plotly_chart(fig2, use_container_width=True)

    # Fraud amount over time
    with col2:
        fraud_txn = txn_df[txn_df["is_fraud"] == 1].copy()
        fraud_txn["date"] = fraud_txn["timestamp"].dt.date
        daily_amount = fraud_txn.groupby("date")["amount"].sum().reset_index()
        fig3 = px.area(
            daily_amount,
            x="date",
            y="amount",
            color_discrete_sequence=["#ef4444"],
            title="Daily Fraudulent Transaction Volume ($)",
            labels={"date": "Date", "amount": "Total Fraud Amount ($)"},
        )
        fig3.update_layout(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font_color="#c9d1d9",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Hourly fraud rate
    hourly = txn_df.groupby("hour")["is_fraud"].agg(["sum", "count"]).reset_index()
    hourly["fraud_rate"] = (hourly["sum"] / hourly["count"] * 100).round(2)
    fig4 = px.bar(
        hourly,
        x="hour",
        y="fraud_rate",
        color="fraud_rate",
        color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
        title="Fraud Rate by Hour of Day (%)",
        labels={"hour": "Hour", "fraud_rate": "Fraud Rate (%)"},
    )
    fig4.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font_color="#c9d1d9",
    )
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — QUANT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📐 Quant Analytics":
    st.markdown(
        """
    <div class="page-header">
        <h1>📐 Quantitative Risk Analytics</h1>
        <p>Statistical risk metrics — z-score, velocity, entropy, cluster density, and risk factor decomposition</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    quant_df = data["quant"]
    scores_df = data["scores"]
    nodes_df = data["nodes"]

    if quant_df is None:
        st.error("Run `quant_features_runner.py` first.")
        st.stop()

    # Merge for plotting
    merged = scores_df.join(quant_df, how="left").fillna(0)
    merged["fraud_label"] = merged["is_fraud"].map({0: "Normal", 1: "Fraud"})

    col1, col2 = st.columns(2)

    # Amount z-score distribution
    with col1:
        fig = px.histogram(
            merged.reset_index(),
            x="amount_zscore",
            color="fraud_label",
            color_discrete_map={"Normal": "#22c55e", "Fraud": "#ef4444"},
            nbins=60,
            barmode="overlay",
            opacity=0.75,
            title="Amount Anomaly Distribution (Z-Score)",
            labels={"amount_zscore": "Amount Z-Score", "count": "Accounts"},
        )
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22", font_color="#c9d1d9"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Transaction velocity distribution
    with col2:
        fig2 = px.histogram(
            merged.reset_index(),
            x="velocity",
            color="fraud_label",
            color_discrete_map={"Normal": "#22c55e", "Fraud": "#ef4444"},
            nbins=60,
            barmode="overlay",
            opacity=0.75,
            title="Transaction Velocity Distribution",
            labels={"velocity": "Velocity (peak/avg)", "count": "Accounts"},
        )
        fig2.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22", font_color="#c9d1d9"
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Entropy scatter
    with col3:
        fig3 = px.scatter(
            merged.reset_index(),
            x="entropy_risk",
            y="risk_score",
            color="fraud_label",
            color_discrete_map={"Normal": "#22c55e", "Fraud": "#ef4444"},
            opacity=0.6,
            title="Recipient Entropy Risk vs. Fraud Risk Score",
            labels={
                "entropy_risk": "Entropy Risk (1=concentrated)",
                "risk_score": "Fraud Risk Score",
            },
            hover_data=["account_id"],
        )
        fig3.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22", font_color="#c9d1d9"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Cluster density
    with col4:
        fig4 = px.scatter(
            merged.reset_index(),
            x="cluster_density",
            y="velocity",
            color="fraud_label",
            color_discrete_map={"Normal": "#22c55e", "Fraud": "#ef4444"},
            opacity=0.6,
            size="risk_score",
            title="Cluster Density vs. Transaction Velocity",
            labels={"cluster_density": "Cluster Density Risk", "velocity": "Velocity"},
            hover_data=["account_id", "risk_score"],
        )
        fig4.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22", font_color="#c9d1d9"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Risk factor breakdown — stacked bar for top 30 accounts
    st.markdown("### 📊 Risk Factor Contribution Breakdown (Top 30)")
    top30 = scores_df.nlargest(30, "risk_score").reset_index()
    components = [
        ("GNN Risk (40%)", "gnn_probability", "#7c3aed"),
        ("Amount Anomaly (20%)", "amount_risk", "#2563eb"),
        ("Velocity Risk (15%)", "velocity_risk", "#059669"),
        ("Centrality Risk (15%)", "centrality_risk", "#d97706"),
        ("Density Risk (10%)", "community_density_risk", "#dc2626"),
    ]
    fig5 = go.Figure()
    for label, col, clr in components:
        if col in top30.columns:
            fig5.add_trace(
                go.Bar(
                    name=label,
                    x=top30["account_id"],
                    y=(
                        top30[col]
                        * [0.4, 0.2, 0.15, 0.15, 0.1][
                            components.index((label, col, clr))
                        ]
                    ),
                    marker_color=clr,
                )
            )
    fig5.update_layout(
        barmode="stack",
        title="Risk Score Factor Decomposition (Top 30 Accounts)",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font_color="#c9d1d9",
        xaxis_tickangle=45,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — ACCOUNT INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Account Inspector":
    st.markdown(
        """
    <div class="page-header">
        <h1>🔍 Account Inspector</h1>
        <p>Deep-dive into any account — SHAP risk drivers + transaction neighbourhood</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    scores_df = data["scores"]
    shap_df = data["shap"]
    shap_v_df = data["shap_v"]
    quant_df = data["quant"]
    nodes_df = data["nodes"]
    txn_df = data["txn"]

    # ── Account selector ──────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 1])
    with col_a:
        account_id = st.text_input("🔎 Enter Account ID", placeholder="e.g. ACC00042")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        show_top = st.button("🎯 Show Random High-Risk Account")

    if show_top and scores_df is not None:
        sample = scores_df[scores_df["risk_score"] >= 70].sample(1)
        account_id = sample.index[0]
        st.info(f"Selected: **{account_id}**")

    if not account_id:
        st.info("Enter an account ID above to inspect it.")
        st.stop()

    if scores_df is not None and account_id not in scores_df.index:
        st.error(f"Account `{account_id}` not found.")
        st.stop()

    row_score = scores_df.loc[account_id]

    # ── Header card ───────────────────────────────────────────────────────────
    risk_val = float(row_score["risk_score"])
    rl = risk_label(risk_val)
    rc = risk_color(risk_val)

    st.markdown(
        f"""
    <div style="background:linear-gradient(135deg,#1a1f2e,#252b3b);border-radius:14px;
                border-left:5px solid {rc};padding:18px 24px;margin-bottom:16px;">
        <h2 style="color:{rc};margin:0">{account_id}</h2>
        <p style="color:#8b949e;margin:4px 0">Community {int(row_score["community_id"])} &bull; 
        {"🚨 Fraud Account" if row_score["is_fraud"] else "✅ Normal Account"}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Score", f"{risk_val:.1f} / 100")
    c2.metric("GNN Risk", f"{row_score['gnn_probability']:.1f}")
    c3.metric("Amount Risk", f"{row_score['amount_risk']:.1f}")
    c4.metric("Velocity Risk", f"{row_score['velocity_risk']:.1f}")
    c5.metric("Centrality Risk", f"{row_score['centrality_risk']:.1f}")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.3])

    # ── SHAP Risk Drivers ─────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 🧠 SHAP Risk Drivers")
        if shap_df is not None and account_id in shap_df.index:
            shap_row = shap_df.loc[account_id]
            for i, col in enumerate(["driver_1", "driver_2", "driver_3"], 1):
                if col in shap_row and shap_row[col]:
                    shap_col = f"driver_{i}_shap"
                    sv = float(shap_row.get(shap_col, 0))
                    direction = "↑ increases risk" if sv > 0 else "↓ reduces risk"
                    color = "#ef4444" if sv > 0 else "#22c55e"
                    st.markdown(
                        f"""
                    <div class="driver-card">
                        <b style="color:{color}">#{i} {shap_row[col]}</b><br>
                        <small style="color:#8b949e">SHAP: {sv:+.4f} — {direction}</small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            # SHAP waterfall (using sub-bar chart)
            if shap_v_df is not None and account_id in shap_v_df.index:
                sv_row = shap_v_df.loc[account_id]
                sv_df = pd.DataFrame(
                    {
                        "Feature": sv_row.index,
                        "SHAP Value": sv_row.values,
                    }
                ).sort_values("SHAP Value", ascending=True)
                sv_df["Color"] = sv_df["SHAP Value"].apply(
                    lambda v: "#ef4444" if v > 0 else "#22c55e"
                )
                fig_shap = go.Figure(
                    go.Bar(
                        x=sv_df["SHAP Value"],
                        y=sv_df["Feature"],
                        orientation="h",
                        marker_color=sv_df["Color"].tolist(),
                    )
                )
                fig_shap.update_layout(
                    title="SHAP Waterfall Chart",
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#161b22",
                    font_color="#c9d1d9",
                    height=360,
                    xaxis_title="SHAP Value (red=increases fraud risk)",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.warning("SHAP data not available. Run `explainability.py` first.")

    # ── Subgraph ──────────────────────────────────────────────────────────────
    with col_right:
        st.markdown("### 🕸️ Transaction Neighbourhood")
        if G is not None and account_id in G:
            hops = st.radio("Hops", [1, 2], horizontal=True, key="inspector_hops")
            with st.spinner("Rendering subgraph …"):
                sub_html = build_pyvis_graph(
                    G,
                    scores_df,
                    subgraph_center=account_id,
                    subgraph_hops=hops,
                )
            st.components.v1.html(sub_html, height=420, scrolling=False)
        else:
            st.warning("Graph not available.")

    # ── Transaction history ───────────────────────────────────────────────────
    st.markdown("### 📋 Transaction History")
    if txn_df is not None:
        acct_txns = txn_df[
            (txn_df["sender_id"] == account_id) | (txn_df["receiver_id"] == account_id)
        ].sort_values("timestamp", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", len(acct_txns))
        c2.metric("Total Volume", f"${acct_txns['amount'].sum():,.0f}")
        c3.metric("Fraud Transactions", int(acct_txns["is_fraud"].sum()))

        st.dataframe(
            acct_txns[
                [
                    "transaction_id",
                    "sender_id",
                    "receiver_id",
                    "amount",
                    "timestamp",
                    "is_fraud",
                ]
            ]
            .head(30)
            .style.apply(
                lambda row: [
                    "background-color:#3b0d0d" if row["is_fraud"] else "" for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
        )
