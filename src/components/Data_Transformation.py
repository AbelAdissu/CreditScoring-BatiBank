import os
import pickle
from dataclasses import dataclass
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import logging

# Configure logging with file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "data_transformation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    raw_data_path = os.path.join(os.getcwd(), 'Data_Sets', 'data.csv')
    transformed_train_data_path = os.path.join(os.getcwd(), 'artifact', 'train_transformed.pkl')
    transformed_test_data_path = os.path.join(os.getcwd(), 'artifact', 'test_transformed.pkl')

class DataTransformation:
    
    def __init__(self, path, threshold=0.01):
        self.path = path
        self.target_column = 'FraudResult'
        self.data_config = DataConfig()
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.smote = SMOTE()

        self.str_replace = [
            'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
            'CustomerId', 'CurrencyCode', 'CountryCode', 'ProviderId', 
            'ProductId', 'ProductCategory', 'ChannelId'
        ]

    def read_data(self):
        """Reads the raw data and splits it for training or testing."""
        try:
            logger.info("Reading raw data from CSV file.")
            self.df = pd.read_csv(self.data_config.raw_data_path)
            
            if 'test' in self.path:
                _, self.df = train_test_split(
                    self.df, test_size=0.2, random_state=42, stratify=self.df[self.target_column]
                )
            else:
                self.df, _ = train_test_split(
                    self.df, test_size=0.2, random_state=42, stratify=self.df[self.target_column]
                )
            logger.info("Data reading and splitting completed.")
            return self.df
        except Exception as e:
            logger.error(f"Error in reading data: {e}", exc_info=True)
            raise

    def drop_columns(self, columns):
        """Drops specified columns from the dataframe."""
        try:
            self.df.drop(columns=columns, inplace=True)
            logger.info(f"Dropped columns: {columns}")
        except Exception as e:
            logger.error(f"Error in dropping columns: {e}", exc_info=True)
            raise

    def replace_strings(self):
        """Removes specific prefixes from string columns and converts to categorical codes."""
        try:
            for column in self.str_replace:
                if column in self.df.columns:
                    self.df[column] = self.df[column].str.replace(f"{column}_", "").astype('category').cat.codes
            logger.info("String replacements completed.")
        except Exception as e:
            logger.error(f"Error in replacing strings: {e}", exc_info=True)
            raise


    def one_hot_encode(self):
        """Applies one-hot encoding to categorical columns."""
        try:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(self.df[['ProductCategory']])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ProductCategory']))
            self.df = pd.concat([self.df.drop('ProductCategory', axis=1), encoded_df], axis=1)
            logger.info("One-hot encoding completed for ProductCategory.")
        except Exception as e:
            logger.error(f"Error in one-hot encoding: {e}", exc_info=True)
            raise

    def add_time_features(self):
        """Extracts date-time features from a timestamp column."""
        try:
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            self.df['Year'] = self.df['TransactionStartTime'].dt.year
            self.df['Month'] = self.df['TransactionStartTime'].dt.month
            self.df['Day'] = self.df['TransactionStartTime'].dt.day
            self.df['Hour'] = self.df['TransactionStartTime'].dt.hour
            self.df['Minute'] = self.df['TransactionStartTime'].dt.minute
            self.df['Second'] = self.df['TransactionStartTime'].dt.second
            logger.info("Time features extracted.")
        except Exception as e:
            logger.error(f"Error in adding time features: {e}", exc_info=True)
            raise
    """
    def feature_selection(self):
        Selects features based on Mutual Information with the target variable.
        try:
            # Drop rows where the target column is NaN
            self.df = self.df.dropna(subset=[self.target_column])

            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
        
            mi_scores = mutual_info_classif(X, y)
            selected_features = pd.Series(mi_scores, index=X.columns).loc[lambda x: x >= self.threshold].index
            self.df = pd.concat([self.df[selected_features], y], axis=1)
            logger.info("Feature selection completed using Mutual Information.")
        except Exception as e:
            logger.error(f"Error in feature selection: {e}", exc_info=True)
            raise
    

    def resample_and_scale(self):
        Applies SMOTE for resampling and StandardScaler for scaling.
        try:
            # Separate features and target
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]

            # Select only numeric columns for scaling
            X_numeric = X.select_dtypes(include=['float64', 'int64'])

            # Scale numeric columns
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)

            # Convert scaled numeric columns back to DataFrame and keep original column names
            X_scaled_df = pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns, index=X.index)

            # Combine scaled numeric columns with non-numeric columns
            X_non_numeric = X.select_dtypes(exclude=['float64', 'int64'])
            X_final = pd.concat([X_scaled_df, X_non_numeric], axis=1)

            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X_final, y)

            # Update the dataframe
            self.df = pd.DataFrame(X_resampled, columns=X_final.columns)
            self.df[self.target_column] = y_resampled

            logger.info("Resampling and scaling completed.")
        except Exception as e:
            logger.error(f"Error in resampling and scaling: {e}", exc_info=True)
            raise

    
    def detect_and_impute_anomalies(self):
        Applies Isolation Forest for anomaly detection and imputes anomalous values.
        try:
            # Exclude the target column and apply Isolation Forest to detect anomalies
            iso_forest = IsolationForest(random_state=42)
            X = self.df.drop(columns=[self.target_column])
        
            # Predict anomalies: -1 for anomalies and 1 for normal data
            anomaly_labels = iso_forest.fit_predict(X)
            self.df['Anomaly'] = anomaly_labels

            # Identify rows where anomalies are detected
            anomaly_indices = self.df[self.df['Anomaly'] == -1].index
        
            # Impute anomalies with the mean or median of the column
            for column in X.columns:
                if self.df[column].dtype in [float, int]:  # Only impute numerical columns
                    median_value = self.df[column].median()
                    self.df.loc[anomaly_indices, column] = median_value

            # Drop the 'Anomaly' column as it’s no longer needed
            self.df.drop(columns=['Anomaly'], inplace=True)
        
            logger.info("Anomaly detection and imputation completed with Isolation Forest.")
        except Exception as e:
            logger.error(f"Error in anomaly detection and imputation: {e}", exc_info=True)
            raise
    """
    def save_transformed_data(self):
        """Saves the transformed dataframe to the specified path."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'wb') as f:
                pickle.dump(self.df, f)
            logger.info(f"Transformed data saved to {self.path}.")
        except Exception as e:
            logger.error(f"Error in saving transformed data: {e}", exc_info=True)
            raise

    def transform(self):
        """Executes the entire transformation pipeline."""
        try:
            logger.info("Starting data transformation process.")
            self.read_data()
            self.drop_columns(['CurrencyCode', 'CountryCode'])
            self.replace_strings()
            self.one_hot_encode()
            self.add_time_features()
            #self.feature_selection()
            #self.resample_and_scale()
            #self.detect_and_impute_anomalies()
            self.save_transformed_data()
            logger.info("Data transformation process completed successfully.")
        except Exception as e:
            logger.error(f"Error in the data transformation process: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    # Initialize transformation objects
    train_transformer = DataTransformation(DataConfig.transformed_train_data_path)
    test_transformer = DataTransformation(DataConfig.transformed_test_data_path)
    
    # Execute transformations
    train_transformer.transform()
    test_transformer.transform()
