
import pandas as pd

file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nChurn counts:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

print("\nTotalCharges blank strings:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())

print("\nDuplicate customer IDs:")
print(df["customerID"].duplicated().sum())

