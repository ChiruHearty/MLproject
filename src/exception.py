import sys  # Used for fetching exception details (like traceback)
from src.logger import logging  # Custom logging utility, assumed to log messages to file or console

# Function to extract detailed error message including file name and line number
def error_message_detail(error, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()  # Get exception type, value, traceback
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename  # File where exception occurred
        line_number = exc_tb.tb_lineno  # Line number where exception occurred
    else:
        file_name = "unknown"
        line_number = "unknown"
    
    # Return formatted error message with file, line, and error info
    return f"Error occurred in python script [{file_name}] at line [{line_number}] with message: {str(error)}"

# Custom exception class that extends base Python Exception
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Call base class constructor
        # Store detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message  # When printed or logged, shows the detailed error

# Testing the CustomException
if __name__ == "__main__":
    try:
        a = 1 / 0  # Intentional ZeroDivisionError
    except Exception as e:
        logging.info("Divided by Zero")  # Log basic info before raising custom error
        raise CustomException(e, sys)  # Raise the exception with full context
    
# Note: 
# 1. Custom Exception Handling (CustomException)
# Purpose:
# Defines a custom exception class that enriches error messages with detailed information 
# about the exact file and line number where the error occurred. This helps in easier debugging 
# during the development of your ML pipeline.
#
# Key Points:
# - Uses Python's sys module to extract traceback details.
# - Formats error messages with filename and line number.
# - Overrides __str__ method to print detailed error info.
# - Example usage shows raising this exception on division by zero.