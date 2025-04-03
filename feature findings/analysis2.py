import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Feature Importance Data (Telecom Churn)
telecom_rf_importance = {
    "Feature": ["MonthlyCharges", "tenure", "TotalCharges", "Contract", "PaymentMethod",
                "OnlineSecurity", "TechSupport", "OnlineBackup", "gender", "PaperlessBilling"],
    "Importance": [0.1765, 0.1750, 0.1677, 0.0820, 0.0512, 0.0480, 0.0431, 0.0278, 0.0275, 0.0261]
}

telecom_permutation_importance = {
    "Feature": ["tenure", "Contract", "InternetService", "MonthlyCharges", "DeviceProtection",
                "TechSupport", "SeniorCitizen", "StreamingTV", "OnlineSecurity", "MultipleLines"],
    "Importance": [0.0439, 0.0284, 0.0089, 0.0083, 0.0072, 0.0072, 0.0061, 0.0055, 0.0050, 0.0041]
}

# Feature Importance Data (Telco Customer Churn)
telco_rf_importance = {
    "Feature": ["total day minutes", "total day charge", "customer service calls",
                "international plan", "total eve charge", "total eve minutes",
                "total intl calls", "total intl minutes", "total intl charge", "total night minutes"],
    "Importance": [0.1364, 0.1323, 0.1190, 0.0776, 0.0698, 0.0637, 0.0461, 0.0436, 0.0421, 0.0367]
}

telco_permutation_importance = {
    "Feature": ["international plan", "total day minutes", "total day charge", "customer service calls",
                "total intl calls", "voice mail plan", "total eve charge", "total eve minutes",
                "total intl minutes", "number vmail messages"],
    "Importance": [0.0496, 0.0438, 0.0429, 0.0381, 0.0208, 0.0186, 0.0169, 0.0118, 0.0118, 0.0096]
}

# Convert to DataFrames
df_telecom_rf = pd.DataFrame(telecom_rf_importance).sort_values(by="Importance", ascending=True)
df_telecom_perm = pd.DataFrame(telecom_permutation_importance).sort_values(by="Importance", ascending=True)
df_telco_rf = pd.DataFrame(telco_rf_importance).sort_values(by="Importance", ascending=True)
df_telco_perm = pd.DataFrame(telco_permutation_importance).sort_values(by="Importance", ascending=True)

# Plot Feature Importance (Telecom Churn)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sns.barplot(data=df_telecom_rf, x="Importance", y="Feature", ax=axes[0, 0], palette="Blues_r")
axes[0, 0].set_title("Telecom Churn - Random Forest Feature Importance")

sns.barplot(data=df_telecom_perm, x="Importance", y="Feature", ax=axes[0, 1], palette="Blues_r")
axes[0, 1].set_title("Telecom Churn - Permutation Feature Importance")

# Plot Feature Importance (Telco Customer Churn)
sns.barplot(data=df_telco_rf, x="Importance", y="Feature", ax=axes[1, 0], palette="Oranges_r")
axes[1, 0].set_title("Telco Customer Churn - Random Forest Feature Importance")

sns.barplot(data=df_telco_perm, x="Importance", y="Feature", ax=axes[1, 1], palette="Oranges_r")
axes[1, 1].set_title("Telco Customer Churn - Permutation Feature Importance")

plt.tight_layout()
plt.show()
