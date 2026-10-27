
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("data/cleaned_telco_churn.csv")

# Chart style
sns.set_theme(style="whitegrid")

# 1. Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/churn_distribution.png")
plt.show()

# 2. Churn rate by contract type
plt.figure(figsize=(8, 5))
contract_churn = pd.crosstab(
    df["Contract"], df["Churn"], normalize="index"
) * 100

contract_churn["Yes"].plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/churn_by_contract.png")
plt.show()

# 3. Monthly charges by churn status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("data/monthly_charges_by_churn.png")
plt.show()

print("All three charts saved inside the data folder!")
