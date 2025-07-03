import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  # Combines pipelines for columns
from sklearn.impute import SimpleImputer  # Fills missing values
from sklearn.pipeline import Pipeline  # Creates sequential transformers
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encoding and feature scaling

from src.exception import CustomException  # Custom error handler
from src.logger import logging  # Custom logging setup
import os

from src.utils import save_object  # Utility to save preprocessor object

@dataclass
class DataTransformationConfig:
    # Path to save the serialized preprocessing object
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

def get_data_transformer_object(self):
    try:
        # Define numerical & categorical features
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = [
            "gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course",
        ]

        # Pipeline for numerical features: impute with median, then scale
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Pipeline for categorical features: impute, encode, scale
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])

        logging.info(f"Categorical columns: {categorical_columns}")
        logging.info(f"Numerical columns: {numerical_columns}")

        # Combine both pipelines using ColumnTransformer
        preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipelines", cat_pipeline, categorical_columns)
        ])

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)

def initiate_data_transformation(self, train_path, test_path):
    try:
        # Read the input CSV files
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Read train and test data completed")

        preprocessing_obj = self.get_data_transformer_object()

        target_column_name = "math_score"
        numerical_columns = ["writing_score", "reading_score"]

        # Split input and target features
        input_feature_train_df = train_df.drop(columns=[target_column_name])
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name])
        target_feature_test_df = test_df[target_column_name]

        logging.info("Applying preprocessing object on training and testing data.")

        # Fit-transform on train, transform on test
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        # Combine features and labels into final NumPy arrays
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        logging.info("Saving preprocessing object.")
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

        # Return the transformed arrays and path to saved preprocessor
        return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

    except Exception as e:
        raise CustomException(e, sys)

# Note:
# Data Transformation Module (DataTransformation class)
# Purpose:
# Constructs preprocessing pipelines for numerical and categorical features, applies transformations 
# on training and test datasets, and saves the preprocessor for later use.
#
# Key Points:
# - Numerical pipeline imputes missing values with median, then scales features.
# - Categorical pipeline imputes most frequent, applies OneHotEncoding, and scales without centering.
# - Combines pipelines via ColumnTransformer.
# - Returns processed NumPy arrays along with preprocessor path.
# - Includes detailed logging and exception handling.
#