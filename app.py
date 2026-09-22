from pathlib import Path

import plotly.express as px
import pandas as pd
import streamlit as st

from src.churn_analysis.pipeline import build_model, load_data, predict_customer

st.set_page_config(page_title="Telecom Churn Intelligence", page_icon="📡", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1400px; padding-top: 2rem; }
    [data-testid="stMetric"] { background: #f5f7fa; border: 1px solid #dfe5ec; padding: 1rem; border-radius: 8px; color: #102a43; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #486581 !important; }
    [data-testid="stMetricValue"] { color: #102a43 !important; }
    [data-testid="stMetricDelta"] { color: #486581 !important; }
    h1, h2, h3 { color: #102a43; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Telecom Churn Intelligence")
st.caption("A practical view of retention risk, customer segments, and next-best actions.")

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload Telco-Customer-Churn.csv", type="csv")
    st.divider()
    st.header("Dashboard guide")
    st.write("Use the filters to isolate segments, then score an individual customer in the action panel.")

try:
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        default_paths = [Path("Telco-Customer-Churn.csv"), Path("data/Telco-Customer-Churn.csv")]
        local_path = next((path for path in default_paths if path.exists()), None)
        if local_path is None:
            st.info("Upload the Telco Customer Churn CSV in the sidebar to open the dashboard.")
            st.stop()
        data = load_data(local_path)
except ValueError as error:
    st.error(str(error))
    st.stop()

model = build_model(data)
with st.sidebar:
    st.header("Segment filters")
    contract_options = sorted(data["Contract"].dropna().unique()) if "Contract" in data else []
    selected_contracts = st.multiselect("Contract", contract_options, default=contract_options)
    min_tenure, max_tenure = int(data["tenure"].min()), int(data["tenure"].max())
    tenure_range = st.slider("Tenure (months)", min_tenure, max_tenure, (min_tenure, max_tenure))

filtered = data[
    data["Contract"].isin(selected_contracts)
    & data["tenure"].between(tenure_range[0], tenure_range[1])
].copy()
churn_rate = filtered["Churn"].eq("Yes").mean() if len(filtered) else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Customers", f"{len(filtered):,}")
metric_cols[1].metric("Churn rate", f"{churn_rate:.1%}")
metric_cols[2].metric("Avg. monthly charge", f"${filtered['MonthlyCharges'].mean():,.2f}" if len(filtered) else "$0.00")
metric_cols[3].metric("Avg. tenure", f"{filtered['tenure'].mean():.1f} mo" if len(filtered) else "0.0 mo")

st.divider()
chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Churn by contract")
    contract_rates = filtered.groupby("Contract", as_index=False).agg(
        **{"Churn rate": ("Churn", lambda values: (values == "Yes").mean())}
    )
    st.plotly_chart(px.bar(contract_rates, x="Contract", y="Churn rate", text_auto=".1%", color="Churn rate", color_continuous_scale="Teal"), use_container_width=True)
with chart_right:
    st.subheader("Monthly charges by outcome")
    st.plotly_chart(px.box(filtered, x="Churn", y="MonthlyCharges", color="Churn", color_discrete_map={"Yes": "#d1495b", "No": "#00798c"}), use_container_width=True)

st.subheader("Retention risk by service")
service_column = "InternetService" if "InternetService" in filtered else "Contract"
service_rates = filtered.groupby(service_column, as_index=False).agg(
    **{"Churn rate": ("Churn", lambda values: (values == "Yes").mean())}
)
st.plotly_chart(px.bar(service_rates, x=service_column, y="Churn rate", text_auto=".1%", color="Churn rate", color_continuous_scale="Sunset"), use_container_width=True)

st.divider()
st.header("Score a customer")
st.caption("Complete the fields below to estimate churn probability using the fitted model.")
with st.form("customer_score"):
    form_columns = st.columns(3)
    tenure = form_columns[0].number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = form_columns[1].number_input("Monthly charges", min_value=0.0, value=70.0, step=5.0)
    total_charges = form_columns[2].number_input("Total charges", min_value=0.0, value=840.0, step=50.0)
    customer = {}
    for column in data.columns:
        if column == "Churn":
            continue
        if column in {"tenure", "MonthlyCharges", "TotalCharges"}:
            customer[column] = {"tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges}[column]
        elif data[column].dtype == "object":
            customer[column] = st.selectbox(column, sorted(data[column].dropna().astype(str).unique()))
        else:
            customer[column] = data[column].median()
    submitted = st.form_submit_button("Calculate churn risk", type="primary")

if submitted:
    result = predict_customer(model, pd.DataFrame([customer])).iloc[0]
    probability = float(result["Churn probability"])
    if probability >= 0.6:
        st.error(f"High risk: {probability:.1%} estimated churn probability. Prioritize a retention offer and proactive outreach.")
    elif probability >= 0.35:
        st.warning(f"Moderate risk: {probability:.1%} estimated churn probability. Monitor engagement and consider an incentive.")
    else:
        st.success(f"Lower risk: {probability:.1%} estimated churn probability. Maintain the current experience.")
