# data_transformation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

def load_data():
    # Load Excel and CSV data
    df_excel = pd.read_excel(r'C:\Users\user\Desktop\end_to_end_ml_project\Data_Sets\data.xlsx')
    df_csv = pd.read_csv(r'C:\Users\user\Desktop\end_to_end_ml_project\Data_Sets\data.csv')
    return df_csv

def clean_data(df):
    # Drop 'CountryCode' and 'CurrencyCode' if they contain single unique values
    if df['CountryCode'].nunique() == 1:
        df.drop('CountryCode', axis=1, inplace=True)
    if df['CurrencyCode'].nunique() == 1:
        df.drop('CurrencyCode', axis=1, inplace=True)

    # Remove text prefixes in selected columns and convert to integers
    columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId', 'ProductId', 'ChannelId']
    for col in columns:
        df[col] = df[col].str.replace(f'{col}_', '').astype(int)
        
    # Convert 'TransactionStartTime' to datetime and extract features
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Year'] = df['TransactionStartTime'].dt.year
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df.drop('TransactionStartTime', axis=1, inplace=True)
    return df

def encode_features(df):
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[['ProductCategory']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ProductCategory']))
    df = df.join(encoded_df).drop('ProductCategory', axis=1)
    return df

def handle_outliers(df):
    # Use DBSCAN and IsolationForest for outlier detection
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['ProductId']])
    df['ProductId_Scaled'] = df_scaled
    isolation = IsolationForest(contamination=0.05)
    df['Outliers'] = isolation.fit_predict(df[['ProductId_Scaled']])
    return df

def select_features(df):
    # Use Mutual Information for feature selection
    X = df.drop(columns=['FraudResult'])
    y = df['FraudResult']
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # Choose features with MI >= 0.01 and retain `FraudResult`
    selected_features = mi_series[mi_series >= 0.01].index.tolist()
    selected_features.append('FraudResult')  # Add 'FraudResult' explicitly
    return df[selected_features]


def balance_data(df):
    # Check if 'FraudResult' exists in the DataFrame
    if 'FraudResult' not in df.columns:
        raise KeyError("The column 'FraudResult' was not found in the DataFrame. Check previous steps for where it may have been removed.")
    
    # Apply SMOTE for class imbalance
    X = df.drop('FraudResult', axis=1)
    y = df['FraudResult']
    
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    
    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled['FraudResult'] = y_res
    return df_resampled



def save_to_pickle(df, file_name="transformed_data.pkl"):
    with open(file_name, 'wb') as file:
        pickle.dump(df, file)
    print(f"DataFrame saved to {file_name}")

def main():
    # Load, clean, and transform data
    df = load_data()
    print("Initial columns:", df.columns)  # Debugging step
    df = clean_data(df)
    print("Columns after clean_data:", df.columns)  # Debugging step
    df = encode_features(df)
    print("Columns after encode_features:", df.columns)  # Debugging step
    df = handle_outliers(df)
    print("Columns after handle_outliers:", df.columns)  # Debugging step
    df = select_features(df)
    print("Columns after select_features:", df.columns)  # Debugging step
    df = balance_data(df)
    print("Columns after balance_data:", df.columns)  # Debugging step

    # Save to pickle
    save_to_pickle(df)

if __name__ == "__main__":
    main()
