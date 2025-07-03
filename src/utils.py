import os  # For file system operations
import sys  # For exception details
import numpy as np
import pandas as pd
import dill  # Advanced pickling module (not used here, but often for serialization)
import pickle  # Python's built-in object serialization
from sklearn.metrics import r2_score  # For regression model evaluation
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from src.exception import CustomException  # Custom exception handler for better error tracing

# Save any Python object (e.g., model) to disk using pickle
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Get directory from path
        os.makedirs(dir_path, exist_ok=True)  # Create directory if not exist

        with open(file_path, "wb") as file_obj:  # Open file in write-binary mode
            pickle.dump(obj, file_obj)  # Serialize and save object
    except Exception as e:
        raise CustomException(e, sys)  # Raise detailed custom exception

# Evaluate multiple models using GridSearchCV and return R² scores
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Dictionary to store model scores

        for i in range(len(list(models))):  # Iterate through each model
            model = list(models.values())[i]  # Get model instance
            para = param[list(models.keys())[i]]  # Get corresponding parameters

            gs = GridSearchCV(model, para, cv=3)  # Grid search with 3-fold CV
            gs.fit(X_train, y_train)  # Fit on training data

            model.set_params(**gs.best_params_)  # Set best params to model
            model.fit(X_train, y_train)  # Re-train model with best params

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)  # R² on train data
            test_model_score = r2_score(y_test, y_test_pred)  # R² on test data

            report[list(models.keys())[i]] = test_model_score  # Save test score
        return report  # Final report of all models
    except Exception as e:
        raise CustomException(e, sys)  # Wrap in custom error

# Load a previously saved object from disk using pickle
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  # Deserialize object
    except Exception as e:
        raise CustomException(e, sys)


# Note:
# Utility Functions (save_object, evaluate_models, load_object)
# Purpose:
# - save_object: Serializes and saves a Python object (e.g., model, preprocessor) using pickle.
# - evaluate_models: Trains multiple models with hyperparameter tuning (via GridSearchCV), 
#   evaluates them using R² score, and returns a report of test scores.
# - load_object: Loads a previously saved object from disk.
#
# Key Points:
# - Exception handling wrapped in custom exceptions for robustness.
# - Uses sklearn utilities for model training and evaluation.
# - Supports saving/loading models or any serializable objects.
#