---
Student Performance Prediction – End-to-End ML Project
This project predicts student academic performance using an end-to-end Machine Learning pipeline built with Python, CatBoost, and Flask. It includes all stages: from data ingestion and preprocessing to model training, evaluation, serialization, and deployment through a web interface.

Project Structure
MLproject/
├── artifacts/
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── catboost_info/
├── logs/
├── data/
│
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md

Key Features
•	Automated data ingestion with train-test split
•	Clean and modular data preprocessing
•	Model training using CatBoostClassifier
•	Model evaluation and serialization (model.pkl)
•	Logging and exception handling
•	Web app using Flask for real-time prediction
•	Fully modular and scalable architecture
________________________________________
Pipeline Details
1. Data Ingestion – data_ingestion.py
•	Loads dataset from data/
•	Splits into train.csv and test.csv
•	Saves output to artifacts/
2. Data Transformation – data_transformation.py
•	Encodes categorical variables
•	Scales numerical features
•	Saves transformation pipeline as preprocessor.pkl
3. Model Training – model_trainer.py
•	Trains a CatBoostClassifier
•	Evaluates accuracy and other metrics
•	Saves model as model.pkl in artifacts/
4. Train Pipeline – train_pipeline.py
•	Orchestrates steps 1–3
5. Prediction Pipeline – predict_pipeline.py
•	Loads model.pkl and preprocessor.pkl
•	Accepts new user input
•	Returns predictions
6. Web App – app.py
•	Flask application to serve predictions
•	Provides HTML form (via templates/) for input
•	Displays predicted result to the user
________________________________________
Utilities & Helpers
•	exception.py: Custom error class to handle and raise exceptions cleanly
•	logger.py: Centralized logging system that tracks pipeline execution
•	utils.py: Common helper functions for saving/loading models, preprocessing objects, etc.
________________________________________
Jupyter Notebooks
•	1. EDA STUDENT PERFORMANCE.ipynb
o	Visualize distributions
o	Correlation matrix
o	Understand data patterns
•	2. MODEL TRAINING.ipynb
o	Train models manually
o	Feature importance plots
o	Evaluation metrics
________________________________________
Key libraries:
•	pandas, numpy
•	scikit-learn
•	catboost
•	flask
•	joblib, os, sys
•	matplotlib, seaborn (for EDA)
________________________________________
Features
•	Robust Data Ingestion: Reads raw CSV data, splits into train/test sets, saves them for downstream processing.
•	Data Transformation Pipelines: Imputation, scaling for numerical features; imputation, one-hot encoding, and scaling for categorical features.
•	Model Training & Tuning: Supports multiple regressors including Random Forest, Gradient Boosting, XGBoost, CatBoost, and more. Uses GridSearchCV for hyperparameter optimization.
•	Evaluation: Selects best model based on R² score; saves the trained model to disk.
•	Prediction Pipeline: Loads saved model and preprocessor; transforms new input data and returns predictions.
•	Flask Web App: User-friendly interface to input student info and receive predicted math score.
________________________________________
Code Highlights
•	Custom Exception Handling: All exceptions are wrapped for detailed error tracing.
•	Logging: Logs are saved with timestamps to both console and file.
•	Reusable Pipelines: Modular design allows easy extension and maintenance.
•	Model Selection: Multiple models tuned and evaluated automatically for best performance.
•	Web Integration: Seamlessly bridges ML backend with an interactive frontend.
