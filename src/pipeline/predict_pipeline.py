import sys
import os
import pandas as pd
from src.exception import CustomException  # Custom error wrapper
from src.utils import load_object  # Utility to load pickled objects

class PredictPipeline:
    def __init__(self):
        pass  # No initialization needed here

    def predict(self, features):
        try:
            # Paths to saved model and preprocessor objects
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            # Load the trained model and preprocessor from disk
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Preprocess input features (e.g., scaling, encoding)
            data_scaled = preprocessor.transform(features)

            # Use the loaded model to predict target values
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            # Raise detailed custom exception if anything fails
            raise CustomException(e, sys)


class CustomData:
    # Initialize input features for prediction
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    # Convert input data to pandas DataFrame for compatibility with pipeline
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            # Create and return a single-row DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise detailed custom exception if conversion fails
            raise CustomException(e, sys)

# Note:
# Prediction Pipeline (PredictPipeline and CustomData classes)
# Purpose:
# - PredictPipeline loads saved preprocessor and model, transforms input features, and returns predictions.
# - CustomData captures raw user inputs, organizes them into a pandas DataFrame suitable for the pipeline.
#
# Key Points:
# - Handles exceptions robustly using the custom exception class.
# - Modular design separates data handling (CustomData) from prediction logic (PredictPipeline).
# - Print statements help debug the loading and prediction steps.
#