M-Pesa Fraud Detector
Project Overview
This project focuses on detecting potentially fraudulent M-Pesa transactions using machine learning and behavioral anomaly detection.

Traditional fraud detection systems often rely on fixed rules and transaction thresholds. However, fraudsters can bypass these rules by structuring transactions into smaller amounts. For example, instead of sending Ksh 147,000 in one transaction, a user could send Ksh 49,000 three times.

This project uses machine learning to identify unusual transaction patterns beyond simple transaction-value thresholds.

Business Problem
Mobile money platforms process large volumes of transactions every day, making it challenging to identify suspicious activity using manual reviews and simple rule-based systems.

Traditional rules may flag transactions based on predefined thresholds, but they can miss more complex patterns involving transaction frequency, timing, location, and behavioral changes.

The objective of this project is to develop an anomaly detection approach that can identify transactions or user behaviors that significantly differ from normal activity.

Data
The project uses transactional data containing information that can be used to identify unusual customer behavior.

Relevant features may include:

Transaction amount

Transaction type

Transaction frequency

Transaction time

Customer or subscriber information

Agent information

Geographic or location information

Historical transaction behavior

The data is prepared and transformed before being used by the machine learning models
Dataset Profile & Feature Engineering

This production pipeline evaluates an independent transactional ledger simulated to match the operational syntax of digital wallet infrastructure in East Africa.

Dataset Name:`mpesa_synthetic_transaction_ledger_v1`
Total Record Volume:** 5,000 corporate and retail transaction logs
Data Source:Synthetically engineered via NumPy exponential and Poisson distributions to mimic high-volume mobile money routing matrixes, integrated with injected zero-day exploit overrides.

Feature Schema & Audit Significance

| Feature Column | Data Type | Analytical & Audit Value |
| :--- | :--- | :--- |
| `txn_id` | String / Object | Unique alphanumeric transaction tracking sequence identifier. |
| `amount_kes` | Float | Transaction gross value in Kenyan Shillings (KES). Monitored against Central Bank of Kenya (CBK) operational transaction caps. |
| `hour_of_day` | Float | Continuous temporal indicator (0.00 - 24.00) mapping exactly when the transfer execution packet was processed. |
| `txn_velocity_1hr` | Integer | Rolling transactional volume indicator tracking actions initiated by the unique entity within a tight 60-minute window. |
| `is_new_agent` | Binary (0/1) | Risk multiplier flag identifying if the cash-out terminal or agent route is fresh, unverified, or geographically inconsistent. |
| `anomaly_score` | Integer | Output ML feature vector. `1` validates standard structural behavior; `-1` isolates high-dimensional risk profiles. |

Engine Performance Diagnostics
Configured with an initial strict contamination boundary of 1.0%, the Isolation Forest model parsed all 5,000 records instantly, flagging exactly **50 high-risk exploits**. 

The engine successfully isolated 100% of the hidden SIM-swap account-drain overrides (characterized by a cluster processing `KES 145,000.00` at `3:15 AM` at an extreme velocity of `12` transactions/hr) alongside natural statistical outliers executing high-velocity movements during illiquid night hours.

Exploratory Data Analysis (EDA)
Exploratory Data Analysis is used to understand transaction patterns and identify potential anomalies before model development.

The analysis focuses on:

Distribution of transaction amounts

Transaction frequency over time

Transaction activity during different hours

Customer transaction behavior

Geographic transaction patterns

Relationships between transaction features

Identification of unusual or extreme observations

Visualizations are used to identify patterns that may not be obvious from individual transactions.

Fraud Indicators
The model considers several behavioral indicators:

Velocity Spikes
A sudden increase in the number of send-money transactions within a rolling 30-minute period may indicate unusual activity.

Unusual Operating Hours
High-value transactions occurring during atypical hours, such as between 2:00 AM and 4:00 AM, can be treated as potential anomalies.

Agent Device Hopping
A subscriber suddenly cashing out through a new or high-risk agent terminal located significantly outside their usual geographic transaction pattern may indicate unusual behavior.

Machine Learning Approach
The project uses the Isolation Forest algorithm for anomaly detection.

Isolation Forest is suitable for this problem because it can identify observations that differ significantly from normal transaction behavior without requiring every fraudulent transaction to be explicitly labeled.

The model considers multiple behavioral characteristics rather than relying on a single transaction-value threshold.

Model Performance
Model performance is evaluated using appropriate anomaly-detection metrics and visualizations.

Depending on the availability of labeled fraud cases, evaluation may include:

Precision

Recall

F1-score

Confusion matrix

Number of detected anomalies

Anomaly score distribution

Model Evaluation & Performance Metrics

To validate the statistical integrity of the Isolation Forest algorithm, the 10 explicitly injected cyber-exploit vectors were isolated as the ground truth fraud class (`1`) against the remaining synthetic ledger profiles (`0`). 

Confusion Matrix Breakdown
True Negatives (TN):4,950 (Standard transactions correctly left unflagged)
False Positives (FP):40 (Benign operational outliers flagged for forensic audit)
False Negatives (FN): 0 (Zero fraud leaks; every single high-risk exploit was captured)
True Positives (TP): 10 (All injected account-drain and structuring attacks successfully caught)

Precision, Recall, & F1-Score

| Evaluation Metric | Score | Risk & Compliance Interpretation |
| :--- | :--- | :--- |
| Recall (Sensitivity) | 100.0%| Zero Fraud Leakage. The model successfully caught every single high-risk malicious vector inside the transaction stream. |
| Precision | 20.0% | Out of 50 total alerts generated, 10 were malicious attacks and 40 were highly irregular but benign transactions requiring manual clearance. |
| F1-Score | 0.333 | Reflects the expected operational behavior of defensive unsupervised models prioritizing complete threat mitigation over false alarm reduction. |

Operational Audit Takeaway 
In active fintech compliance ecosystems, a 100% Recall Rate is the critical threshold metric because the financial penalty and regulatory liability of a missed exploit (False Negative) heavily outweigh the cost of an auditor reviewing a false alarm (False Positive). The 40 statistical anomalies flagged by the system represent high-velocity, off-hour liquidity movements that naturally justify a standard Suspicious Activity Report (SAR) review queue.

Results
The model identifies transactions that exhibit behavioral patterns significantly different from the expected transaction profile.

The results can be used to:

Identify potentially suspicious transactions

Prioritize transactions for further investigation

Detect unusual customer behavior

Complement traditional rule-based fraud detection

Reduce reliance on fixed transaction thresholds

Key Findings & Analytical Insights

 Anomaly Distribution Analysis
Classification Rate: Exactly 1.0% of the entire transactional dataset (50 out of 5,000 logs) was classified as anomalous under a baseline `0.01` contamination strategy.
Exploit Verification: 
  20% (10 records): Linked directly to the manually injected zero-day exploit overrides (SIM-swap account drains and velocity structuring).
  80% (40 records): Natural statistical exceptions isolated natively by the algorithm. These represent retail profiles processing erratic transaction volumes during illiquid resting hours (e.g., `TXN_00531` pushing `KES 20,431.11` at `3:20 AM` at a velocity of `5`), mimicking compromised credentials or active unauthorized device attachments.

Feature Isolation Hierarchy
While an unsupervised Isolation Forest segregates profiles across multi-dimensional spaces rather than static coefficients, evaluating the path lengths across the generated trees reveals the primary drivers of anomaly classification:

1. `txn_velocity_1hr` (Primary Driver): High-frequency repetitions within a rolling 60-minute window triggered the fastest branch isolations. Normal user distributions cluster around a Poisson mean of ~1.2, making the exploit value of 12 transactions/hour
instantly identifiable.
2. `hour_of_day` & `amount_kes` (Interaction Isolation): High value alone or late-night executions alone do not trigger automatic isolation. The engine captures the non-linear intersection of maximum daily parameters (`KES 145,000.00`) executing simultaneously during dormant structural hours (`3:15 AM`).
3. `is_new_agent` (Risk Multiplier): Operates as a critical weights vector, severely decreasing isolation path depths when an unverified or fresh terminal state overlaps with sudden transactional volume shifts.

Core Dashboard Visualizations

Multidimensional Risk Clusters (Interactive Plotly Scatter): Maps `Amount_KES` directly against `Hourly_Velocity` with color assignment determined by `Risk_Status`. It visually partitions the high-density normal customer cluster (rendered in green below KES 20,000) away from high-threat vectors (rendered in red stretching out into the top right operational boundaries).
Live Fraud Inspector Queue: An embedded interactive structural matrix that surfaces high-risk outliers sorted descending by value. This queue provides instant operational visibility to forensic auditors, prioritizing corporate financial exposure risk.
CBK Statutory Alert Strip: A dynamic threshold compliance warning strip built explicitly to match Central Bank of Kenya transactional ceiling guidelines, identifying immediate regulatory filing requirements.


Future Improvements
Future development could improve the system by:

Testing additional anomaly detection algorithms such as Local Outlier Factor and One-Class SVM

Comparing Isolation Forest with supervised classification models when labeled fraud data is available

Incorporating real-time transaction monitoring

Adding more detailed geographic and device-level behavioral features

Developing automated fraud alerts

Performing model retraining as new transaction patterns emerge

Evaluating the model using a larger and more representative dataset

Deploying the model as an API or real-time fraud detection service

Technologies Used
Python

Jupyter Notebook

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Git

GitHub

