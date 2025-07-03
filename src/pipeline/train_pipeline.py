# train_pipeline.py

import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("Training pipeline started.")

        # Step 1: Data Ingestion
        logging.info("Initiating Data Ingestion process...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")

        # Step 2: Data Transformation
        logging.info("Initiating Data Transformation process...")
        data_transformation = DataTransformation()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data Transformation completed.")

        # Step 3: Model Training
        logging.info("Initiating Model Training process...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Model Training completed with R2 Score: {r2_score}")

    except Exception as e:
        logging.error("Exception occurred in training pipeline.")
        raise CustomException(e, sys)

# Note:
#Runs the full training lifecycle:
# -Reads and splits data
# -Transforms input features using pipelines
# -Trains multiple regression models with hyperparameter tuning
# -Saves the best-performing model

#Logs every step for traceability
#Raises CustomException for any failure with line-level debug info