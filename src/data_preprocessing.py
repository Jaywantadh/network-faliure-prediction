import pandas as pd

# Load datasets
file_path_1 = "../data/Telco-Customer-Churn.csv"
file_path_2 = "../data/telecom_churn.csv"

def load_data(file_path):
    """Loads dataset and displays basic info."""
    df = pd.read_csv(file_path)
    print("Dataset Loaded: ", file_path)
    print("Columns: ", df.columns)
    return df

# Load both datasets
df1 = load_data(file_path_1)
df2 = load_data(file_path_2)

# Drop unnecessary columns
drop_cols = ['customerID', 'phone number', 'Unnamed: 0']  # Modify as needed
df1 = df1.drop(columns=[col for col in drop_cols if col in df1.columns], errors='ignore')
df2 = df2.drop(columns=[col for col in drop_cols if col in df2.columns], errors='ignore')

# Rename churn column to failure_event if present
if 'churn' in df1.columns:
    df1.rename(columns={'churn': 'failure_event'}, inplace=True)
if 'Churn' in df2.columns:
    df2.rename(columns={'Churn': 'failure_event'}, inplace=True)

# Convert categorical values to numerical (e.g., Yes -> 1, No -> 0)
categorical_cols = ['failure_event', 'gender', 'Partner', 'Dependents', 'PhoneService']  # Modify as needed
for col in categorical_cols:
    if col in df1.columns:
        df1[col] = df1[col].astype('category').cat.codes
    if col in df2.columns:
        df2[col] = df2[col].astype('category').cat.codes

# Display final dataset info
print("Final Processed Data Info:")
print(df1.info())
print(df2.info())

# Save processed data
df1.to_csv("processed_telecom_churn.csv", index=False)
df2.to_csv("processed_telco_customer_churn.csv", index=False)
