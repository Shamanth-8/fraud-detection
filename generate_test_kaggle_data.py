import pandas as pd
import numpy as np

def generate_kaggle_style_dataset(filepath="test_fraud_dataset.csv", n_rows=5000, fraud_ratio=0.04):
    print(f"Generating {n_rows} rows of test data...")
    
    # Generate Time
    time_col = np.linspace(0, 100000, n_rows)
    
    # Generate V1 to V28
    np.random.seed(99) # different seed from earlier
    v_cols = {f"V{i}": np.random.normal(0, 1.5, n_rows) for i in range(1, 29)}
    
    # Generate Amount (log-normal distribution typical for amounts)
    amount_col = np.random.lognormal(mean=2, sigma=1.5, size=n_rows)
    
    # Generate Class labels (0 = genuine, 1 = fraud)
    class_col = np.random.choice([0, 1], size=n_rows, p=[1 - fraud_ratio, fraud_ratio])
    
    # Embed some patterns so that our model can "detect" some correlation
    # E.g. If class is 1, V1 and V2 get shifted, and amount tends to be higher
    fraud_indices = np.where(class_col == 1)[0]
    
    v_cols["V1"][fraud_indices] += np.random.normal(-3, 1, len(fraud_indices))
    v_cols["V2"][fraud_indices] += np.random.normal(3, 1, len(fraud_indices))
    v_cols["V3"][fraud_indices] += np.random.normal(-5, 2, len(fraud_indices))
    v_cols["V4"][fraud_indices] += np.random.normal(4, 1.5, len(fraud_indices))
    
    # Tweak amounts for fraud
    amount_col[fraud_indices] = np.random.uniform(500, 3000, len(fraud_indices))
    
    # Assemble DataFrame
    df = pd.DataFrame({"Time": time_col})
    for col_name, data in v_cols.items():
        df[col_name] = data
    
    df["Amount"] = amount_col
    df["Class"] = class_col
    
    df.to_csv(filepath, index=False)
    print(f"✅ Successfully created {filepath} ({len(df)} rows, {df['Class'].sum()} fraud cases)")

if __name__ == "__main__":
    generate_kaggle_style_dataset()
