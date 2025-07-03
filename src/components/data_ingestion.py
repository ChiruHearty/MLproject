import os
import sys
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Custom logging setup
import pandas as pd  # For data handling

from sklearn.model_selection import train_test_split  # For splitting dataset
from dataclasses import dataclass  # For creating simple data classes

# Importing transformation and model training modules
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Data class to store file paths for train, test, and raw data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Main Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initialize config

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load data from source CSV
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create directory to save artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split data into train/test sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train/test data to files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return file paths for next steps
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)  # Raise wrapped exception for better debugging

# Run all steps in sequence
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()  # Step 1: Ingest data

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)  # Step 2: Transform data

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))  # Step 3: Train model and print result

# Note:
# Data Ingestion Module (DataIngestion class)
# Purpose:
# Reads raw data CSV, splits it into train and test datasets, saves these datasets to disk, 
# and returns file paths for downstream processing.
#
# Key Points:
# - Uses pandas for reading CSV and splitting data.
# - Saves raw, train, and test datasets into an artifacts folder.
# - Logs progress at key steps for traceability.
# - Uses train_test_split with a fixed random state for reproducibility.
#  