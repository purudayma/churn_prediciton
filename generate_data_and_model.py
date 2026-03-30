"""
E-Commerce Customer Churn Prediction
Data Generation + Model Training Script
Run this FIRST before starting the Flask app.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

# ─── 1. Generate Synthetic Dataset ────────────────────────────────────────────

print("📦 Generating synthetic dataset (50,000 rows)...")

np.random.seed(42)
N = 50_000

# Basic demographics
customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, N + 1)]
ages = np.random.normal(38, 12, N).clip(18, 75).astype(int)
genders = np.random.choice(["Male", "Female"], N, p=[0.48, 0.52])
locations = np.random.choice(
    ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"],
    N, p=[0.18, 0.17, 0.16, 0.12, 0.12, 0.10, 0.08, 0.07]
)
payment_methods = np.random.choice(
    ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"],
    N, p=[0.30, 0.25, 0.25, 0.12, 0.08]
)

# Behavioural features
tenure = np.random.exponential(24, N).clip(1, 84).astype(int)
number_of_orders = (tenure * np.random.uniform(0.5, 3.0, N)).clip(1, 300).astype(int)
avg_order_value = np.random.lognormal(7.5, 0.6, N).clip(150, 15000).round(2)
total_spending = (number_of_orders * avg_order_value * np.random.uniform(0.8, 1.2, N)).round(2)
last_purchase_days_ago = np.random.exponential(45, N).clip(1, 365).astype(int)
discount_used = np.random.choice([0, 1], N, p=[0.40, 0.60])
satisfaction_score = np.random.choice(range(1, 11), N,
    p=[0.04, 0.05, 0.07, 0.09, 0.11, 0.14, 0.16, 0.14, 0.11, 0.09])
app_usage_time = np.random.exponential(30, N).clip(0, 180).round(1)
customer_support_calls = np.random.poisson(1.5, N).clip(0, 15)
returned_orders = (number_of_orders * np.random.beta(1.2, 8, N)).astype(int)

# ─── Churn Logic (realistic patterns) ─────────────────────────────────────────
churn_score = (
    0.30 * (last_purchase_days_ago / 365)          # High inactivity → churn
    - 0.20 * (satisfaction_score / 10)              # Low satisfaction → churn
    + 0.15 * (customer_support_calls / 15)          # Many support calls → churn
    - 0.10 * (app_usage_time / 180)                 # Low app usage → churn
    + 0.10 * (returned_orders / (number_of_orders + 1))  # High returns → churn
    - 0.05 * (tenure / 84)                          # Long tenure → less churn
    + 0.05 * (1 - discount_used * 0.5)              # No discounts → slight churn
    + np.random.normal(0, 0.08, N)                  # Noise
)
churn_prob = 1 / (1 + np.exp(-3 * (churn_score - 0.3)))
churn = (np.random.uniform(0, 1, N) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Age": ages,
    "Gender": genders,
    "Location": locations,
    "Tenure": tenure,
    "Number_of_Orders": number_of_orders,
    "Average_Order_Value": avg_order_value,
    "Total_Spending": total_spending,
    "Last_Purchase_Days_Ago": last_purchase_days_ago,
    "Payment_Method": payment_methods,
    "Discount_Used": discount_used,
    "Satisfaction_Score": satisfaction_score,
    "App_Usage_Time": app_usage_time,
    "Customer_Support_Calls": customer_support_calls,
    "Returned_Orders": returned_orders,
    "Churn": churn_label,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/ecommerce_churn.csv", index=False)
print(f"✅ Dataset saved → data/ecommerce_churn.csv  | Churn rate: {churn.mean():.1%}")

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────

print("\n🔧 Preprocessing data...")

df2 = df.drop(columns=["CustomerID"]).copy()

# Encode target
df2["Churn"] = (df2["Churn"] == "Yes").astype(int)

# Encode categoricals
le_gender = LabelEncoder()
le_location = LabelEncoder()
le_payment = LabelEncoder()
df2["Gender"] = le_gender.fit_transform(df2["Gender"])
df2["Location"] = le_location.fit_transform(df2["Location"])
df2["Payment_Method"] = le_payment.fit_transform(df2["Payment_Method"])

# Feature engineering
df2["Orders_Per_Month"] = df2["Number_of_Orders"] / (df2["Tenure"] + 1)
df2["Avg_Spend_Per_Day"] = df2["Total_Spending"] / (df2["Last_Purchase_Days_Ago"] + 1)
df2["Return_Rate"] = df2["Returned_Orders"] / (df2["Number_of_Orders"] + 1)

feature_cols = [c for c in df2.columns if c != "Churn"]
X = df2[feature_cols]
y = df2["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ─── 3. Train Models ──────────────────────────────────────────────────────────

print("\n🤖 Training models...")

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_sc, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))
print(f"   Logistic Regression Accuracy: {lr_acc:.4f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"   Random Forest Accuracy:       {rf_acc:.4f}")

# Use best model
best_model = rf if rf_acc >= lr_acc else lr
best_name = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\n🏆 Best model: {best_name} ({max(rf_acc, lr_acc):.4f})")

# ─── 4. Save Artifacts ────────────────────────────────────────────────────────

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_cols, "model/feature_cols.pkl")
joblib.dump({
    "le_gender": le_gender,
    "le_location": le_location,
    "le_payment": le_payment
}, "model/encoders.pkl")

# Save dashboard stats
churn_by_loc = df.groupby("Location")["Churn"].apply(lambda x: (x == "Yes").mean()).round(3).to_dict()
churn_by_gender = df.groupby("Gender")["Churn"].apply(lambda x: (x == "Yes").mean()).round(3).to_dict()
avg_satisfaction = df.groupby("Churn")["Satisfaction_Score"].mean().round(2).to_dict()
payment_dist = df["Payment_Method"].value_counts().to_dict()

stats = {
    "total_customers": int(N),
    "churn_rate": float(round(churn.mean(), 4)),
    "churn_count": int(churn.sum()),
    "no_churn_count": int(N - churn.sum()),
    "model_accuracy": float(round(max(rf_acc, lr_acc), 4)),
    "best_model": best_name,
    "avg_age": float(round(ages.mean(), 1)),
    "avg_tenure": float(round(tenure.mean(), 1)),
    "avg_satisfaction": float(round(satisfaction_score.mean(), 2)),
    "churn_by_location": churn_by_loc,
    "churn_by_gender": churn_by_gender,
    "payment_distribution": payment_dist,
    "feature_importance": dict(zip(feature_cols, rf.feature_importances_.round(4).tolist()))
}

with open("model/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✅ All artifacts saved:")
print("   model/churn_model.pkl")
print("   model/scaler.pkl")
print("   model/feature_cols.pkl")
print("   model/encoders.pkl")
print("   model/stats.json")
print("\n🚀 Setup complete! Run: python app.py")
