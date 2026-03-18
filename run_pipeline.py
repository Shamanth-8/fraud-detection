#!/usr/bin/env python3
"""
run_pipeline.py
---------------
One-shot runner: executes the full fraud detection pipeline end-to-end.
Run this instead of each script individually.

Usage:
    python run_pipeline.py [--skip-gnn]
"""

import subprocess
import sys
import os
import argparse
import time

STEPS = [
    (
        "Data Generation",
        "pipeline/graph_data_generator.py",
        "artifacts/graph_transactions.csv",
    ),
    ("Graph Analytics", "pipeline/graph_builder.py", "artifacts/node_features.csv"),
    (
        "Quant Features",
        "pipeline/quant_features_runner.py",
        "artifacts/quant_features.csv",
    ),
    ("GNN Training", "pipeline/gnn_model.py", "artifacts/gnn_predictions.csv"),
    ("Risk Scoring", "pipeline/risk_scorer.py", "artifacts/risk_scores.csv"),
    ("Explainability", "pipeline/explainability.py", "artifacts/shap_explanations.csv"),
]


def run_step(name: str, script: str, output: str, skip: bool = False):
    if skip:
        print(f"\n⏭️  Skipping {name}")
        return True
    if os.path.exists(output):
        print(f"\n✅ {name} already done ({output} exists). Skipping.")
        return True
    print(f"\n{'=' * 60}")
    print(f"▶  {name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    ret = subprocess.run([sys.executable, script], check=False)
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"\n❌ {name} FAILED (exit code {ret.returncode})")
        return False
    print(f"✅ {name} completed in {dt:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-gnn",
        action="store_true",
        help="Skip GNN training (uses fraud label as proxy)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run all steps even if output exists"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Graph-Based Financial Fraud Detection Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.force:
        for f in [
            "artifacts/graph_transactions.csv",
            "artifacts/node_features.csv",
            "artifacts/quant_features.csv",
            "artifacts/gnn_predictions.csv",
            "artifacts/risk_scores.csv",
            "artifacts/shap_explanations.csv",
            "artifacts/shap_values.csv",
            "artifacts/transaction_graph.gpickle",
            "artifacts/gnn_fraud_model.pt",
        ]:
            if os.path.exists(f):
                os.remove(f)
        print("🗑️  Cleared previous outputs (--force)")

    for name, script, output in STEPS:
        skip = args.skip_gnn and script == "gnn_model.py"
        ok = run_step(name, script, output, skip=skip)
        if not ok:
            print(f"\n🛑 Pipeline aborted at: {name}")
            sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   🎉 Pipeline complete! Launch the dashboard:            ║")
    print("║       streamlit run dashboard/app.py                      ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
