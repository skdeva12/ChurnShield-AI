
import pandas as pd

# Load the original dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove duplicate customer records
df = df.drop_duplicates(subset="customerID")

# Fill missing charges for zero-tenure customers
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Save cleaned dataset
df.to_csv("data/cleaned_telco_churn.csv", index=False)

# Verify the cleaned dataset
print("Cleaned dataset shape:", df.shape)

print("\nMissing values remaining:")
print(df.isnull().sum().sum())

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nFirst five cleaned rows:")
print(df.head())

print("\nCleaned dataset saved successfully!")
