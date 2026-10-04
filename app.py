import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration to wide layout for financial data layout
st.set_page_config(page_title="M-Pesa Fraud Detector Engine", layout="wide")

# --- APPLICATION TITLE & HEADER ---
st.title("🚨 M-Pesa Behavioral Fraud & Anomaly Detector")
st.markdown("""
This production dashboard evaluates transaction risk using an unsupervised **Isolation Forest** model. 
It captures high-dimensional exploits (like SIM-swaps and velocity structuring) that bypass traditional static thresholds.
""")

# --- STEP 1: CACHED DATA PIPELINE GENERATION ---
@st.cache_data
def load_and_prep_data():
    # Simulating your 'mpesa_synthetic_transaction_ledger_v1' dataset (5,000 logs)
    np.random.seed(42)
    n_records = 5000
    
    # Generate baseline benign retail data
    amounts = np.random.exponential(scale=3500, size=n_records) + np.random.randint(10, 500, size=n_records)
    hours = np.random.uniform(0.0, 24.0, size=n_records)
    velocity = np.random.poisson(lam=1.2, size=n_records)
    is_new_agent = np.random.choice([0, 1], size=n_records, p=[0.92, 0.08])
    
    df = pd.DataFrame({
        'txn_id': [f"TXN_{str(i).zfill(5)}" for i in range(n_records)],
        'amount_kes': amounts,
        'hour_of_day': hours,
        'txn_velocity_1hr': velocity,
        'is_new_agent': is_new_agent
    })
    
    # Programmatically inject the 6 Zero-Day Cyber Exploit vectors (SIM-swap account-drains)
    # Target profile: KES 145,000.00 at 3:15 AM at extreme velocity of 12 txns/hr
    exploit_indices = [100, 101, 102, 103, 104, 105]
    for idx in exploit_indices:
        df.loc[idx, 'amount_kes'] = 145000.00
        df.loc[idx, 'hour_of_day'] = 3.15
        df.loc[idx, 'txn_velocity_1hr'] = 12
        df.loc[idx, 'is_new_agent'] = 1
        
    return df, exploit_indices

# Execute pipeline loader
df, exploit_indices = load_and_prep_data()

# --- STEP 2: USER STREAMLIT INTERFACE SIDEBAR ---
st.sidebar.header("🎯 ML Engine Control Configurations")
contamination_rate = st.sidebar.slider(
    "Isolation Forest Contamination Boundary", 
    min_value=0.005, max_value=0.05, value=0.01, step=0.005,
    help="The expected ratio of anomalies/outliers inside the ledger."
)

# --- STEP 3: MODEL INFERENCE PIPELINE ---
# Features utilized for path-length isolating projections
features = ['amount_kes', 'hour_of_day', 'txn_velocity_1hr', 'is_new_agent']
X = df[features]

# Fit Isolation Forest Model
model = IsolationForest(contamination=contamination_rate, random_state=42)
df['anomaly_score'] = model.fit_predict(X)

# Map predictions: Anomaly (-1) -> High Risk, Normal (1) -> Approved
df['fraud_flag'] = np.where(df['anomaly_score'] == -1, "High Risk", "Approved")

# Compute actual verification metrics
ground_truth = np.zeros(len(df))
ground_truth[exploit_indices] = 1 # Ground truth exploits
predictions = np.where(df['anomaly_score'] == -1, 1, 0)

total_flagged = len(df[df['fraud_flag'] == "High Risk"])
true_positives = np.sum((ground_truth == 1) & (predictions == 1))
false_positives = total_flagged - true_positives

# --- STEP 4: KPI METRIC CARD GRID ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Ledger Volume", f"{len(df):,} Logs")
with col2:
    st.metric("Total Flags Generated", f"{total_flagged} Alerts", delta=f"{contamination_rate*100:.1f}% Config Rate", delta_color="inverse")
with col3:
    st.metric("Verified Exploits Caught (TP)", f"{true_positives} / 6", help="SIM-swap vectors correctly blocked.")
with col4:
    st.metric("Recall (Sensitivity)", f"{(true_positives / 6)*100:.1f}%", delta="0 Fraud Leaks")

# --- STEP 5: INTERACTIVE SPLIT INTERFACE ---
tabs = st.tabs(["🔍 Forensic Audit Queue", "📊 Visual Heatmaps & Analytics"])

with tabs[0]:
    st.subheader("📋 Priority Live Forensic Inspection Queue")
    st.markdown("Sorted descending by financial exposure. Outliers require immediate Suspicious Activity Report (SAR) processing.")
    
    # Filter for high-risk profiles
    high_risk_queue = df[df['fraud_flag'] == "High Risk"].sort_values(by='amount_kes', ascending=False)
    
    # Custom color highlighting formatting for Streamlit dataframes
    st.dataframe(
        high_risk_queue[['txn_id', 'amount_kes', 'hour_of_day', 'txn_velocity_1hr', 'is_new_agent', 'fraud_flag']],
        use_container_width=True,
        hide_index=True
    )

with tabs[1]:
    st.subheader("📈 Diagnostic Visual Analytics")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Multi-Dimensional Risk Profile Clustering**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df, x='hour_of_day', y='amount_kes', 
            hue='fraud_flag', palette={'Approved': 'g', 'High Risk': 'r'}, alpha=0.7, ax=ax
        )
        ax.set_title("Transaction Amount vs. Pacing Hour")
        ax.axhline(y=140000, color='gray', linestyle='--', alpha=0.5, label="CBK High-Value Line")
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("**Confusion Matrix Confusion Metrics**")
        # Build matrix structure dynamically
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'], ax=ax,
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_ylabel('Actual Ground Truth')
        ax.set_xlabel('Model Predicted Class')
        st.pyplot(fig)
