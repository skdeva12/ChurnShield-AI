
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_telco_churn.csv")

# Basic dataset information
print("Dataset shape:", df.shape)

print("\nColumn information:")
print(df.info())

# Churn distribution
print("\nChurn counts:")
print(df["Churn"].value_counts())

# Churn percentages
print("\nChurn percentages:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

# Churn by contract type
print("\nChurn by contract type:")
print(pd.crosstab(df["Contract"], df["Churn"]))

# Churn by tenure group
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

print("\nChurn by tenure group:")
print(pd.crosstab(df["TenureGroup"], df["Churn"]))

# Average monthly charges by churn status
print("\nAverage monthly charges:")
print(df.groupby("Churn", observed=True)["MonthlyCharges"].mean().round(2))
