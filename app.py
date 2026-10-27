
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="ChurnShield AI",
    page_icon="🛡️",
    layout="wide"
)

# Professional dashboard styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0b1120;
    }

    [data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.8);
    }

    h1 {
        color: #e2e8f0;
        font-weight: 800;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #e2e8f0;
        font-weight: 700;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #172554, #111827);
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    [data-testid="stMetricValue"] {
        color: #60a5fa;
        font-weight: 800;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    div[data-testid="stForm"] {
        background-color: #111827;
        border: 1px solid #263449;
        padding: 22px;
        border-radius: 16px;
    }

    hr {
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("data/random_forest_model.pkl")

model = load_model()

# Dashboard header
st.title("🛡️ ChurnShield AI")
st.subheader("Customer Churn Prediction & Retention Intelligence")
st.write(
    "Predict customer churn risk using a machine learning model "
    "and identify customers who may need retention support."
)

st.divider()

# Customer input form
st.header("Customer Risk Assessment")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 6)

    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.subheader("Charges")
    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=90.0, step=1.0
        )

    with col5:
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=540.0, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    probability = model.predict_proba(customer)[0, 1]
    threshold = 0.35

    st.divider()
    st.header("Prediction Results")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Estimated Churn Probability", f"{probability:.1%}")

    with metric2:
        st.metric("Selected Threshold", f"{threshold:.0%}")

    st.progress(float(probability))

    if probability >= threshold:
        st.error("Higher churn risk — consider reviewing retention options.")
        st.write(
            "Suggested action: review customer concerns, service quality, "
            "contract options, and whether suitable support can improve retention."
        )
    else:
        st.success("Lower churn risk under the selected threshold.")

    st.caption(
        "This is a machine-learning estimate, not a guarantee. "
        "Predictions depend on the model and the information entered."
    )
    
st.divider()

st.header("📊 Customer Churn Analytics")

# Load the cleaned dataset
@st.cache_data
def load_customer_data():
    return pd.read_csv("data/cleaned_telco_churn.csv")

df = load_customer_data()

# Key metrics
total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
churn_rate = (churned_customers / total_customers) * 100
retained_customers = total_customers - churned_customers

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Churn Rate", f"{churn_rate:.2f}%")


st.subheader("📉 Churn Rate by Contract Type")

contract_data = (
    pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
)

contract_chart = contract_data[["Yes"]].rename(
    columns={"Yes": "Churn Rate (%)"}
)

st.bar_chart(
    contract_chart,
    horizontal=True
)


st.subheader("📅 Average Customer Tenure")

tenure_data = df.groupby("Churn")["tenure"].mean().round(1)

tenure_data.index = tenure_data.index.map({
    "No": "Retained Customers",
    "Yes": "Churned Customers"
})

st.bar_chart(
    tenure_data,
    horizontal=True
)

st.caption(
    "Analytics are calculated from the cleaned Telco Customer Churn dataset."
)


st.divider()


st.divider()
st.header("🎯 Random Forest Confusion Matrix")

st.write(
    "Compare the model's predictions with the actual customer outcomes."
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

@st.cache_data
def get_confusion_matrix_data():
    data = pd.read_csv("data/cleaned_telco_churn.csv")

    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_test, y_test

X_test_cm, y_test_cm = get_confusion_matrix_data()

y_pred_cm = model.predict(X_test_cm)

cm = confusion_matrix(y_test_cm, y_pred_cm)


fig, ax = plt.subplots(figsize=(7, 5))

# Match the dashboard's dark theme
fig.patch.set_facecolor("#0b1120")
ax.set_facecolor("#111827")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

# Improve text and axis styling
ax.set_title(
    "Actual vs Predicted Customer Churn",
    color="#e2e8f0",
    fontsize=14,
    pad=15
)
ax.set_xlabel("Predicted Label", color="#cbd5e1", fontsize=11)
ax.set_ylabel("Actual Label", color="#cbd5e1", fontsize=11)

ax.tick_params(axis="both", colors="#cbd5e1")

# Make cell numbers readable
for text in disp.text_.ravel():
    text.set_fontsize(14)
    text.set_fontweight("bold")

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Rows represent actual outcomes; columns represent predicted outcomes. "
    "False negatives are customers who churned but were predicted to stay."
)

st.write(
    "Explore which customer features the Random Forest model "
    "uses most when making churn predictions."
)

# Access the fitted preprocessing and Random Forest steps
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get the feature names after categorical encoding
feature_names = preprocessor.get_feature_names_out()

# Build feature-importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
})

# Make feature names easier to read
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("categorical__", "", regex=False)
    .str.replace("remainder__", "", regex=False)
)

# Show the 10 most important features
importance_df = importance_df.sort_values(
    by="Importance", ascending=False
).head(10)

st.subheader("Top 10 Predictive Features")

st.bar_chart(
    importance_df.set_index("Feature")["Importance"]
)

st.caption(
    "Feature importance measures how much the fitted Random Forest "
    "relies on features when making predictions. It does not prove "
    "that a feature causes churn or explain an individual prediction."
)

