"""
graph_data_generator.py
-----------------------
Generates a synthetic graph-compatible transaction dataset.
Output: graph_transactions.csv
Columns: transaction_id, sender_id, receiver_id, amount, timestamp, is_fraud
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ─── Config ────────────────────────────────────────────────────────────────────
SEED = 42
N_ACCOUNTS = 2000
N_TRANSACTIONS = 20000
N_FRAUD_RINGS = 12  # tightly-connected rings
RING_SIZE = (6, 18)  # min/max accounts per ring
FRAUD_RATE = 0.03  # ~3% fraud transactions
START_DATE = datetime(2024, 1, 1)
DURATION_DAYS = 90
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "graph_transactions.csv")

random.seed(SEED)
np.random.seed(SEED)

print("Generating synthetic transaction graph …")

# ─── 1. Account pool ───────────────────────────────────────────────────────────
accounts = [f"ACC{str(i).zfill(5)}" for i in range(N_ACCOUNTS)]

# ─── 2. Fraud rings ────────────────────────────────────────────────────────────
fraud_rings = []
fraud_accounts = set()
for ring_id in range(N_FRAUD_RINGS):
    ring_size = random.randint(*RING_SIZE)
    available = [a for a in accounts if a not in fraud_accounts]
    if len(available) < ring_size:
        break
    ring = random.sample(available, ring_size)
    fraud_rings.append(ring)
    fraud_accounts.update(ring)

fraud_account_list = list(fraud_accounts)
print(f"  → {N_FRAUD_RINGS} fraud rings covering {len(fraud_account_list)} accounts")

# ─── 3. Generate transactions ──────────────────────────────────────────────────
records = []
n_fraud_target = int(N_TRANSACTIONS * FRAUD_RATE)
n_normal = N_TRANSACTIONS - n_fraud_target


def random_timestamp(burst=False):
    """Return a random timestamp; burst=True clusters within short windows."""
    if burst:
        # Burst: pick a random day and cluster within a 2-hour window
        day_offset = random.randint(0, DURATION_DAYS - 1)
        hour_offset = random.uniform(0, 22)
        minute_offset = random.uniform(0, 120)
        return START_DATE + timedelta(
            days=day_offset, hours=hour_offset, minutes=minute_offset
        )
    else:
        total_minutes = DURATION_DAYS * 24 * 60
        return START_DATE + timedelta(minutes=random.uniform(0, total_minutes))


txn_id = 0

# Normal transactions
normal_accounts = [a for a in accounts if a not in fraud_accounts]
for _ in range(n_normal):
    sender = random.choice(normal_accounts)
    receiver = random.choice([a for a in accounts if a != sender])
    amount = round(np.random.lognormal(mean=3.5, sigma=1.2), 2)  # realistic amounts
    ts = random_timestamp(burst=False)
    records.append(
        {
            "transaction_id": f"TXN{str(txn_id).zfill(7)}",
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "timestamp": ts,
            "is_fraud": 0,
        }
    )
    txn_id += 1

# Fraud transactions (within and between rings — burst pattern)
for _ in range(n_fraud_target):
    ring = random.choice(fraud_rings)
    sender = random.choice(ring)
    # 70% intra-ring, 30% to outside (money exit)
    if random.random() < 0.70:
        candidates = [a for a in ring if a != sender]
        receiver = random.choice(candidates) if candidates else random.choice(accounts)
    else:
        receiver = random.choice(accounts)
    # Fraud: either very small (layering) or very large (laundering)
    if random.random() < 0.5:
        amount = round(np.random.uniform(1, 50), 2)
    else:
        amount = round(np.random.uniform(5000, 50000), 2)
    ts = random_timestamp(burst=True)  # Fraud clusters in bursts
    records.append(
        {
            "transaction_id": f"TXN{str(txn_id).zfill(7)}",
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "timestamp": ts,
            "is_fraud": 1,
        }
    )
    txn_id += 1

# ─── 4. Shuffle and save ───────────────────────────────────────────────────────
df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"  → {len(df):,} transactions saved → {OUTPUT_FILE}")
print(f"  → Fraud: {df['is_fraud'].sum():,} ({100 * df['is_fraud'].mean():.1f}%)")
print(f"  → Unique senders:   {df['sender_id'].nunique():,}")
print(f"  → Unique receivers: {df['receiver_id'].nunique():,}")
print(f"  → Amount range: ${df['amount'].min():.2f} – ${df['amount'].max():,.2f}")
print("Done.")
