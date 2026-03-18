# Graph-Based Financial Fraud Detection & Quantitative Risk Analytics

> End-to-end **Graph ML pipeline** — synthetic data → graph construction → GNN-based fraud classification → quantitative risk scoring → interactive Streamlit dashboard.

---

## Architecture

```
graph_data_generator → graph_builder → quant_features → gnn_model → risk_scorer → dashboard
       │                    │               │                │             │            │
       ▼                    ▼               ▼                ▼             ▼            ▼
 graph_transactions   node_features   quant_features   gnn_model.pt   risk_scores   Streamlit
      .csv                .csv             .csv                        .csv          6 pages
```

### Core Components

| Component | Description |
|---|---|
| **graph_data_generator.py** | Generates ~20,000 synthetic transactions across ~2,000 accounts with embedded fraud rings (~3% fraud rate) |
| **graph_builder.py** | Builds a directed NetworkX graph, computes 5 centrality metrics + Louvain community detection |
| **quant_features.py** | Computes quantitative risk metrics: amount z-score, transaction velocity, Shannon entropy, cluster density |
| **gnn_model.py** | 3-layer GraphSAGE (PyTorch Geometric) for node-level fraud classification with class-balanced training |
| **risk_scorer.py** | Weighted composite risk score (0–100) combining GNN probability, amount/velocity/centrality/density risk |
| **explainability.py** | SHAP-based feature importance via surrogate Random Forest — top-3 risk drivers per account |
| **app.py** | 6-page Streamlit dashboard: Overview, Network, Fraud Rings, Temporal, Quant Analytics, Account Inspector |

---

## Project Structure

```
project/
├── pipeline/                       # ML pipeline modules
│   ├── graph_data_generator.py
│   ├── graph_builder.py
│   ├── quant_features.py
│   ├── quant_features_runner.py
│   ├── gnn_model.py
│   ├── risk_scorer.py
│   └── explainability.py
├── artifacts/                      # Generated data & model files
│   ├── graph_transactions.csv
│   ├── node_features.csv
│   ├── quant_features.csv
│   ├── gnn_fraud_model.pt
│   ├── gnn_predictions.csv
│   ├── risk_scores.csv
│   ├── shap_explanations.csv
│   └── shap_values.csv
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── run_pipeline.py                 # One-shot pipeline runner
├── requirements.txt
└── README.md
```

---

## Installation

### Dashboard only (lightweight)

```bash
pip install -r requirements.txt
```

### Full ML pipeline (includes PyTorch, TensorFlow)

```bash
pip install -r requirements-pipeline.txt
```

> **Note**: PyTorch and PyTorch Geometric may require platform-specific install commands. See [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/) docs.

---

## Usage

### Run the Full Pipeline

```bash
python run_pipeline.py          # runs all 6 steps sequentially
python run_pipeline.py --force  # re-runs everything from scratch
python run_pipeline.py --skip-gnn  # skip GNN training (uses fraud labels as proxy)
```

### Run Individual Modules

```bash
python pipeline/graph_data_generator.py    # → artifacts/graph_transactions.csv
python pipeline/graph_builder.py           # → artifacts/node_features.csv
python pipeline/quant_features_runner.py   # → artifacts/quant_features.csv
python pipeline/gnn_model.py               # → artifacts/gnn_fraud_model.pt
python pipeline/risk_scorer.py             # → artifacts/risk_scores.csv
python pipeline/explainability.py          # → artifacts/shap_explanations.csv
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** to explore:

| Page | What It Shows |
|---|---|
| **Fraud Overview** | KPI cards, risk score histogram, fraud vs normal pie chart, top-20 risky accounts |
| **Transaction Network** | Interactive PyVis graph, node coloring by risk, subgraph drill-down |
| **Fraud Rings** | Louvain community clusters, scatter plots, suspicious community table |
| **Temporal Analysis** | Daily transaction volume, fraud burst heatmap, hourly fraud rate |
| **Quant Analytics** | Z-score distribution, velocity, entropy scatter, risk factor decomposition |
| **Account Inspector** | Per-account SHAP risk drivers, waterfall chart, transaction neighbourhood graph |

---

## 🚀 Deploy to Streamlit Cloud

1. Push the repo to **GitHub** (artifacts CSVs included, heavy model files excluded via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** and select:
   - **Repository**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Click **Deploy** — Streamlit Cloud will install from `requirements.txt` automatically

> The dashboard works fully without PyTorch/GNN — it reads pre-computed CSV artifacts and can also process uploaded datasets in-browser.

---

## Key Technologies

- **Graph ML**: NetworkX, PyTorch Geometric (GraphSAGE)
- **Quantitative Analytics**: Z-score, Shannon entropy, transaction velocity, cluster density
- **Explainability**: SHAP (TreeExplainer on surrogate Random Forest)
- **Visualization**: Streamlit, Plotly, PyVis
- **Community Detection**: Louvain algorithm (python-louvain)
