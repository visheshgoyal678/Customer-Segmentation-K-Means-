"""
Generates a synthetic retail customer dataset with realistic underlying
segments (so clustering has genuine structure to recover), then shuffles
and saves it as data/customers.csv.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

def make_segment(n, age_range, income_range, spend_range, freq_range, tenure_range, genders):
    return pd.DataFrame({
        "Age": np.random.randint(age_range[0], age_range[1], n),
        "AnnualIncome_k": np.round(np.random.uniform(income_range[0], income_range[1], n), 1),
        "SpendingScore": np.round(np.random.uniform(spend_range[0], spend_range[1], n), 1),
        "PurchaseFrequency": np.round(np.random.uniform(freq_range[0], freq_range[1], n), 1),
        "TenureMonths": np.random.randint(tenure_range[0], tenure_range[1], n),
        "Gender": np.random.choice(genders, n),
    })

segments = [
    # Young, low income, high spending score (impulse spenders)
    make_segment(90, (18, 30), (15, 40), (60, 95), (8, 20), (1, 24), ["Male", "Female"]),
    # High income, high spending (premium/loyal customers)
    make_segment(80, (28, 45), (70, 140), (65, 95), (12, 25), (6, 60), ["Male", "Female"]),
    # High income, low spending (careful/frugal affluent)
    make_segment(85, (35, 60), (70, 140), (5, 35), (1, 6), (1, 48), ["Male", "Female"]),
    # Low income, low spending (budget conscious)
    make_segment(95, (30, 65), (15, 40), (5, 35), (1, 5), (1, 36), ["Male", "Female"]),
    # Mid income, mid spending (average / general population)
    make_segment(150, (25, 55), (40, 70), (35, 65), (5, 12), (1, 60), ["Male", "Female"]),
]

df = pd.concat(segments, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "CustomerID", [f"CUST{i:05d}" for i in range(1, len(df) + 1)])

# add a little realistic noise / outliers
noise_idx = np.random.choice(df.index, 15, replace=False)
df.loc[noise_idx, "SpendingScore"] = np.clip(
    df.loc[noise_idx, "SpendingScore"] + np.random.normal(0, 20, 15), 0, 100
)

out_path = "/home/claude/customer_segmentation/data/customers.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head())
