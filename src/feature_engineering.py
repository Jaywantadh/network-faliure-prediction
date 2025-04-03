import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

print(os.path.exists('../data/processed_telecom_churn.csv'))
print(os.path.exists('../data/processed_telco_customer_churn.csv'))

teleco_churn = pd.read_csv('../data/processed_telecom_churn.csv')
teleco_customer = pd.read_csv('../data/processed_telco_customer_churn.csv')


def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace = True)
            else:
                df[col].fillna(df[col].median(), inplace = True)
    return df

teleco_churn = handle_missing_values(teleco_churn)
teleco_customer = handle_missing_values(teleco_customer)

print(teleco_churn.head())
print(teleco_customer.head())

coloumns_to_drop = ['customerID', 'Unnamed: 0']
teleco_churn.drop(columns = [col for col in coloumns_to_drop if col in teleco_churn.columns], inplace = True)
teleco_customer.drop(columns = [col for col in coloumns_to_drop if col in teleco_customer.columns], inplace = True)

label_enc = LabelEncoder()

categeorical_coloumns = teleco_churn.select_dtypes(include = ['object']).columns

for col in categeorical_coloumns:
    teleco_churn[col] = label_enc.fit_transform(teleco_churn[col])

categeorical_coloumns_customer = teleco_customer.select_dtypes(include =['object']).columns

for col in categeorical_coloumns_customer:
    teleco_customer[col] = label_enc.fit_transform(teleco_customer[col])

scaler = StandardScaler()

numerical_coloumns = teleco_churn.select_dtypes(include = ['int64', 'float64']).columns
teleco_churn[numerical_coloumns] = scaler.fit_transform(teleco_churn[numerical_coloumns])

numerical_coloumns_customer = teleco_customer.select_dtypes(include = ['int64', 'float64']).columns
teleco_customer[numerical_coloumns_customer] = scaler.fit_transform(teleco_customer[numerical_coloumns_customer])

teleco_churn.to_csv('../data/processed_telecom_churn.csv', index=False)
teleco_customer.to_csv('../data/processed_telco_customer_churn.csv', index=False)

print("Feature Engineeering Completed: Processed file has been saved.")

