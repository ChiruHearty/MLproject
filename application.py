from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# Import your prediction pipeline and input data wrapper
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application  # Alias for AWS or other platform compatibility

# Route for the home page (landing page)
@app.route('/')
def index():
    return render_template('index.html')  # Render a welcome/index HTML page

# Route to handle prediction form submission and display result
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If accessed via GET (e.g., browser URL), show input form page
        return render_template('home.html')
    else:
        # On POST (form submission), capture user inputs from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            # NOTE: swapped reading_score and writing_score from form inputs? Check below
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        
        # Convert input data into DataFrame expected by the pipeline
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Initialize prediction pipeline, run prediction on input data
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render the same input form page, now showing the prediction result
        return render_template('home.html', results=results[0])

# Run Flask app on all interfaces at port 5000 with debug enabled
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# Note:
# Flask Web App for Prediction
# Purpose:
# A simple Flask web application exposing an interface for users to input feature data and get predictions rendered in a web page.
#
# Key Points:
# - / route serves a landing page (index.html).
# - /predictdata supports GET (show form) and POST (process form and predict).
# - User inputs are collected via Flask request.form, converted to CustomData, then predicted.
# - Results are passed back to home.html for display.
# - Minor issue: reading and writing scores swapped during form input extraction.
# - Debug prints included to trace execution.
#