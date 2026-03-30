"""
E-Commerce Customer Churn Prediction — Flask App
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# ─── Load model artifacts ─────────────────────────────────────────────────────
MODEL_DIR = "model"

model       = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
scaler      = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
encoders    = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "stats.json")) as f:
    STATS = json.load(f)

# Label maps (must match LabelEncoder training order)
GENDER_CLASSES   = list(encoders["le_gender"].classes_)
LOCATION_CLASSES = list(encoders["le_location"].classes_)
PAYMENT_CLASSES  = list(encoders["le_payment"].classes_)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", stats=STATS)


@app.route("/predict")
def predict_page():
    return render_template("predict.html",
        genders=GENDER_CLASSES,
        locations=LOCATION_CLASSES,
        payments=PAYMENT_CLASSES
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.form

        # Raw inputs
        age                   = float(d["age"])
        gender_str            = d["gender"]
        location_str          = d["location"]
        tenure                = float(d["tenure"])
        number_of_orders      = float(d["number_of_orders"])
        avg_order_value       = float(d["avg_order_value"])
        total_spending        = float(d["total_spending"])
        last_purchase_days    = float(d["last_purchase_days_ago"])
        payment_str           = d["payment_method"]
        discount_used         = float(d["discount_used"])
        satisfaction_score    = float(d["satisfaction_score"])
        app_usage_time        = float(d["app_usage_time"])
        customer_support_calls = float(d["customer_support_calls"])
        returned_orders       = float(d["returned_orders"])

        # Encode categoricals
        gender_enc   = encoders["le_gender"].transform([gender_str])[0]
        location_enc = encoders["le_location"].transform([location_str])[0]
        payment_enc  = encoders["le_payment"].transform([payment_str])[0]

        # Derived features
        orders_per_month  = number_of_orders / (tenure + 1)
        avg_spend_per_day = total_spending / (last_purchase_days + 1)
        return_rate       = returned_orders / (number_of_orders + 1)

        # Build feature vector in training order
        feat = {
            "Age": age,
            "Gender": gender_enc,
            "Location": location_enc,
            "Tenure": tenure,
            "Number_of_Orders": number_of_orders,
            "Average_Order_Value": avg_order_value,
            "Total_Spending": total_spending,
            "Last_Purchase_Days_Ago": last_purchase_days,
            "Payment_Method": payment_enc,
            "Discount_Used": discount_used,
            "Satisfaction_Score": satisfaction_score,
            "App_Usage_Time": app_usage_time,
            "Customer_Support_Calls": customer_support_calls,
            "Returned_Orders": returned_orders,
            "Orders_Per_Month": orders_per_month,
            "Avg_Spend_Per_Day": avg_spend_per_day,
            "Return_Rate": return_rate,
        }

        import pandas as pd
        X = pd.DataFrame([[feat[c] for c in feature_cols]], columns=feature_cols)

        # Random Forest doesn't need scaling; apply only if model is LR
        if "Logistic" in STATS["best_model"]:
            X = scaler.transform(X)

        prediction   = model.predict(X)[0]
        probability  = model.predict_proba(X)[0]
        churn_prob   = float(round(probability[1] * 100, 1))
        no_churn_prob = float(round(probability[0] * 100, 1))
        result_label = "Churn" if prediction == 1 else "No Churn"

        # Risk level
        if churn_prob >= 70:
            risk = "High Risk"
            risk_color = "danger"
        elif churn_prob >= 40:
            risk = "Medium Risk"
            risk_color = "warning"
        else:
            risk = "Low Risk"
            risk_color = "success"

        return render_template("result.html",
            result=result_label,
            churn_prob=churn_prob,
            no_churn_prob=no_churn_prob,
            risk=risk,
            risk_color=risk_color,
            inputs={
                "Age": int(age),
                "Gender": gender_str,
                "Location": location_str,
                "Tenure": int(tenure),
                "Orders": int(number_of_orders),
                "Avg Order Value": f"₹{avg_order_value:,.0f}",
                "Total Spending": f"₹{total_spending:,.0f}",
                "Last Purchase": f"{int(last_purchase_days)} days ago",
                "Payment": payment_str,
                "Satisfaction": f"{int(satisfaction_score)}/10",
                "App Usage": f"{app_usage_time} min",
                "Support Calls": int(customer_support_calls),
                "Returned Orders": int(returned_orders),
            }
        )

    except Exception as e:
        return render_template("predict.html",
            error=f"Prediction error: {str(e)}",
            genders=GENDER_CLASSES,
            locations=LOCATION_CLASSES,
            payments=PAYMENT_CLASSES
        )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=STATS)


@app.route("/api/stats")
def api_stats():
    return jsonify(STATS)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run()
