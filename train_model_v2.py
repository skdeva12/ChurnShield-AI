
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load the original cleaned dataset
df = pd.read_csv("data/cleaned_telco_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

# Scale numeric features and encode categories
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    ("categorical", 
     __import__("sklearn").preprocessing.OneHotEncoder(
         handle_unknown="ignore",
         drop="first"
     ),
     categorical_columns)
])

# Build the full pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Split data before fitting to prevent data leakage
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train and evaluate
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("LOGISTIC REGRESSION V2")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Churn", "Churn"]
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "data/logistic_regression_v2.pkl")
print("\nImproved pipeline saved!")
