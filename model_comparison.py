
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Model Comparison | ChurnShield AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ChurnShield AI — Model Comparison")
st.write(
    "Compare Logistic Regression and Random Forest "
    "on the same customer churn test set."
)

# Metrics measured during your previous model runs
results = pd.DataFrame({
    "Model": [
        "Logistic Regression V2",
        "Random Forest"
    ],
    "Accuracy": [0.8070, 0.7835],
    "ROC-AUC": [0.8422, 0.8351],
    "Churn Precision": [0.66, 0.58],
    "Churn Recall": [0.56, 0.65],
    "Churn F1-score": [0.61, 0.62]
})

st.subheader("Performance Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Best Accuracy", "80.70%", "Logistic Regression")
col2.metric("Best ROC-AUC", "0.8422", "Logistic Regression")
col3.metric("Best Churn Recall", "65%", "Random Forest")

st.divider()

st.subheader("Metrics by Model")

metric = st.selectbox(
    "Choose a metric",
    [
        "Accuracy",
        "ROC-AUC",
        "Churn Precision",
        "Churn Recall",
        "Churn F1-score"
    ]
)

fig = px.bar(
    results,
    x="Model",
    y=metric,
    color="Model",
    text=results[metric].map(lambda x: f"{x:.2%}" if x <= 1 else str(x)),
    title=f"{metric} Comparison",
    range_y=[0, 1]
)

fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Full Results Table")

st.dataframe(
    results.style.format({
        "Accuracy": "{:.2%}",
        "ROC-AUC": "{:.4f}",
        "Churn Precision": "{:.2%}",
        "Churn Recall": "{:.2%}",
        "Churn F1-score": "{:.2f}"
    }),
    use_container_width=True
)

st.subheader("Which model should we choose?")

st.markdown("""
- **Logistic Regression V2:** Better accuracy and ROC-AUC in these results.
- **Random Forest:** Better churn recall and churn F1-score.
- **Business decision:** Random Forest may be preferable when missing an at-risk
  customer is especially costly. Logistic Regression may be preferable when
  overall discrimination and accuracy are priorities.

The final choice should depend on retention costs and the desired balance
between missed churners and false alarms.
""")

st.caption(
    "These values are from the earlier evaluation runs. "
    "They are recorded results, not automatically recalculated metrics."
)
