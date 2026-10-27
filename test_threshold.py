
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

df = pd.read_csv("data/cleaned_telco_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("data/random_forest_model.pkl")
probabilities = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

for threshold in [0.50, 0.40, 0.35, 0.30]:
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n{'=' * 35}")
    print("THRESHOLD:", threshold)
    print(classification_report(
        y_test, predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
