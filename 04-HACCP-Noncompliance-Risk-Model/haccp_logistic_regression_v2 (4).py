"""HACCP Noncompliance Risk Modeling — Python version
Niusha Bagheri | Logistic Regression + ROC + Risk by Area

Run:
  python haccp_logistic_regression.py

Requires:
  pandas, numpy, scikit-learn, matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv("haccp_compliance_regression_wafercream_2023_2024.csv")

X = df.drop(columns=["Noncompliance"])
y = df["Noncompliance"]

cat = ["Area", "Shift", "Sampling_Reason", "Product"]
num = ["Humidity_pct", "Temp_C"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", "passthrough", num)
])

model = Pipeline([
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=2000))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
model.fit(Xtr, ytr)
probs = model.predict_proba(Xte)[:, 1]

auc = roc_auc_score(yte, probs)
print(f"AUC: {auc:.3f}")

# ROC
fpr, tpr, _ = roc_curve(yte, probs)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={auc:.2f}) — HACCP Noncompliance Risk Model")
plt.savefig("haccp_roc_curve_py.png", dpi=150, bbox_inches="tight")

# Risk by area (avg predicted probability using all data)
all_prob = model.predict_proba(X)[:, 1]
risk_area = pd.DataFrame({"Area": df["Area"], "prob": all_prob}).groupby("Area")["prob"].mean().sort_values()

plt.figure()
plt.barh(risk_area.index, risk_area.values)
plt.xlabel("Avg Predicted Risk")
plt.title("Average Predicted Noncompliance Risk by Area")
plt.savefig("haccp_risk_by_area_py.png", dpi=150, bbox_inches="tight")

print("Saved: haccp_roc_curve_py.png, haccp_risk_by_area_py.png")
