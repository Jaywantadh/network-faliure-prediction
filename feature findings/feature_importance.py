import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

# Load Processed Data
telecom_churn = pd.read_csv("../data/processed_telecom_churn.csv")
telco_customer_churn = pd.read_csv("../data/processed_telco_customer_churn.csv")

# Function to analyze feature importance
def analyze_feature_importance(df, dataset_name):
    print(f"\n{'='*40}\nFeature Importance Analysis for {dataset_name}\n{'='*40}")
    
    # Ensure 'Churn' column exists
    if "Churn" not in df.columns:
        raise ValueError(f"Churn column not found in {dataset_name} dataset!")

    # Split Features & Target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype('int')

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # ====== 1. Correlation Analysis ======
    corr = df.corr()["Churn"].drop("Churn").sort_values(ascending=False)
    print("\nTop Correlated Features with Churn:\n", corr.head(10))

    # Visualizing Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, fmt=".2f")
    plt.title(f"Feature Correlation Heatmap - {dataset_name}")
    plt.show()

    # ====== 2. Feature Importance from Random Forest ======
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    print("\nRandom Forest Feature Importance:\n", feature_importance.head(10))

    # Visualizing Feature Importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10), palette="viridis", legend=False)
    plt.title(f"Top 10 Feature Importance - {dataset_name} (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()

    # ====== 3. Permutation Importance ======
    perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"Feature": X.columns, "Importance": perm_importance.importances_mean})
    perm_df = perm_df.sort_values(by="Importance", ascending=False)

    print("\nPermutation Importance:\n", perm_df.head(10))

    # Visualizing Permutation Importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=perm_df.head(10), palette="magma", legend=False)
    plt.title(f"Top 10 Permutation Importance - {dataset_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()

    # ====== 4. SHAP Values Analysis ======
    explainer = shap.Explainer(rf_model, X_train)
    shap_values = explainer(X_test, check_additivity=False)

    print("\nSHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test)

# Run analysis on both datasets
analyze_feature_importance(telecom_churn, "Telecom Churn")
analyze_feature_importance(telco_customer_churn, "Telco Customer Churn")
