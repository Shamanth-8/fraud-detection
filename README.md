# 🕵️ Graph-Based Financial Fraud Detection & Quantitative Risk Analytics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-deep_learning_on_graphs-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An end-to-end, production-oriented **Graph ML pipeline** designed to detect sophisticated financial fraud rings. We ingest raw transaction data, construct relational graphs, apply state-of-the-art **GraphSAGE** neural networks, compute **quantitative risk metrics**, and surface actionable insights through an interactive **Streamlit dashboard**.

### 🌟 High-Performance Model Evaluation
We don't just build models; we rigorously evaluate them. Our GraphSAGE model significantly outperforms traditional tabular baselines (like Random Forests) by leveraging topological network structures, excelling in imbalanced data scenarios.

Both models were evaluated on the exact same train/test split. For fraud detection, **AUC-PR (Area Under the Precision-Recall Curve)** is the most meaningful metric due to high class imbalance.

> 📝 **Note to Reviewers:** These exceptionally high precision and recall metrics reflect performance on a **synthetic fraud dataset** specifically generated to test architectural pathways. Be aware that these metrics establish the model framework's validity and robustness, but performance will vary naturally on noisy, real-world data.

#### Performance Metrics

| Metric | GraphSAGE | Baseline (Random Forest) |
|--------|-----------|--------------------------|
| **AUC-PR** | **0.9655** | 0.7416 |
| AUC-ROC | 0.9782 | 0.8207 |
| F1 Score (Macro) | 0.9608 | 0.8584 |
| F1 Score (Weighted)| 0.9797 | 0.9314 |
| Precision (Fraud) | 0.9655 | 0.9744 |
| Recall (Fraud) | 0.9032 | 0.6129 |

*Visual plots such as PR curves, ROC curves, Confusion Matrices, and SHAP Feature Importances are generated into the `results/plots/` directory during evaluation.*

---

## 🚀 Why This Matters (The Technical Edge)

Most traditional fraud detection models treat transactions identically, ignoring the complex, hidden networks formed by bad actors. This project solves that by bridging **Graph Representation Learning** with **Quantitative Finance Analytics**:

* 🧠 **For ML Engineers:** Implements a custom **3-layer GraphSAGE** architecture via PyTorch Geometric. We tackle extreme class imbalance (~3% fraud rate) natively during training and utilize **SHAP** via surrogate TreeExplainers to conquer the notorious "black box" problem of neural networks.
* 📈 **For Quant Analysts:** Beyond deep learning, the pipeline computes robust statistical features. It applies **Z-score anomaly detection** on transfer amounts, measures **transaction velocity**, calculates **Shannon Entropy** to detect concentrated fund dispersals, and utilizes the **Louvain heuristic** for community density metrics.
* 💻 **For Python Developers:** Built with a hyper-modular, decoupled architecture. Components are cleanly separated, state is managed via explicit artifact generation, and the entire pipeline can be executed synchronously using a clean `run_pipeline.py` orchestrator.

---

## 🏗️ Architecture & Data Tapestry

```text
graph_data_generator → graph_builder → quant_features → gnn_model → risk_scorer → dashboard
       │                    │               │                │             │            │
       ▼                    ▼               ▼                ▼             ▼            ▼
 graph_transactions   node_features   quant_features   gnn_model.pt   risk_scores   Streamlit
      .csv                .csv             .csv                        .csv          6 pages
```

### 🧩 Core Modules

| Component | Engineering Purpose |
|---|---|
| **`graph_data_generator.py`** | Synthesizes complex adversarial datasets (~20k txns, ~2k accounts) with embedded topology (bipartite structures, cyclical fraud rings). |
| **`graph_builder.py`** | Ingests edges, builds directed NetworkX graphs. Computes *PageRank, Betweenness, Eigenvector & Degree Centralities*, plus community detection. |
| **`quant_features.py`** | Vectorized pandas workflows computing purely statistical/quantitative risk vectors (velocity, entropy, density). |
| **`gnn_model.py`** | Deep Graph Neural Network. Learns spatial embeddings from localized account neighborhoods to predict illicit behavior. |
| **`risk_scorer.py`** | An ensemble-like compositor outputting a calibrated score (0-100) weighting network logic, ML probability, and quant anomalies. |
| **`explainability.py`** | Trains interpretable surrogate models (Random Forests) on node features to derive local **SHAP values**, explaining *why* an account is risky. |

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Graph-Based-Financial-Fraud-Detection.git
   cd Graph-Based-Financial-Fraud-Detection
   ```

2. **Choose your environment:**

   * **A. Dashboard Only (Lightweight)** - Best if you just want to run the UI utilizing pre-computed artifacts.
     ```bash
     pip install -r requirements.txt
     ```
   
   * **B. Full ML Pipeline (Heavy)** - Required to train GraphSAGE and run the orchestrator.
     ```bash
     pip install -r requirements-pipeline.txt
     ```
     > *Note for ML Devs:* PyTorch & PyTorch Geometric are hardware-specific. Check [PyTorch](https://pytorch.org/) docs to maximize CUDA/MPS acceleration.

---

## 💻 Usage & Orchestration

### 1. Execute the ML Pipeline
Our pipeline orchestrator handles the DAG sequence gracefully, tracking artifact generation and logging execution times.
```bash
python run_pipeline.py             # Standard sequential execution
python run_pipeline.py --force     # Nukes artifacts and rebuilds from scratch
python run_pipeline.py --skip-gnn  # Skips PyTorch training (uses proxy labels)
```

*(You can also run modules individually. e.g., `python pipeline/gnn_model.py`)*

### 2. Standalone Model Evaluation
Generates precision-recall & ROC plots, confusion matrices, and calculates macro metrics comparing the deep learning network against a Tree-based standard.
```bash
python evaluate_model.py
```

### 3. Launch the Intelligence Dashboard
Fire up the Streamlit frontend to visualize computations.
```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Modules (What You'll See)

Open **http://localhost:8501** to explore 6 distinct intelligence lenses:

1. **Fraud Overview:** High-level executive KPIs, distribution mappings, and top-tier risky accounts instantly surfaced.
2. **Transaction Network:** Interactive, physics-based `PyVis` graphs. Zoom into local node neighborhoods and trace fund flows.
3. **Fraud Rings:** Advanced cluster visualization utilizing the Louvain algorithm. Identify synchronized account ecosystems.
4. **Temporal Analysis:** Time-series aggregations, heatmaps of transactional bursts, and chronological anomaly charting.
5. **Quant Analytics:** Deep-dive into z-scores, transaction velocities, and feature decomposers bypassing black-box ML.
6. **Account Inspector:** The ultimate investigator tool. See waterfall **SHAP** charts dictating the exact variables driving an account's risk score.

---

## 🚀 Deployed to Streamlit Cloud

**Live Demo:** [https://fraud-detection-graphs.streamlit.app/](https://fraud-detection-graphs.streamlit.app/)

The UI is natively optimized for zero-fuss deployment to Streamlit Community Cloud:
1. Push to GitHub (ensure large `.pt` models are explicitly git-ignored to respect limits; the app utilizes proxy fallback logic without them).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo, and point the main file to `dashboard/app.py`.
3. Deploy!

The dashboard natively supports dynamic, in-browser CSV uploads ensuring scalability to custom datasets.

---

**Built with:** _NetworkX, PyTorch Geometric, SHAP, Pandas, Plotly, PyVis, Scikit-Learn, & Streamlit._
