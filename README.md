'

# M-Pesa Fraud Detector

## Project Overview
This project applies machine learning specifically an Isolation Forest model—to detect anomalous and potentially fraudulent M-Pesa transactions beyond static thresholds. 

It analyzes 5,000 synthetic transaction records to flag behavioral patterns like velocity spikes and unusual operating hours.

## Key Metrics & Performance
Total Records Parsed: 5,000

Contamination Rate: 1.0% (50 high-risk alerts generated)

Recall (Sensitivity): 100.0% (Successfully caught all injected malicious vectors)

Precision: 12.0% (6 malicious attacks vs. 44 benign outliers)
F1-Score: 0.21