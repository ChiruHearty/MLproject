import logging  # Built-in module for logging messages
import os  # For file path and directory operations
from datetime import datetime  # To timestamp log file names

# Define a directory to store logs; create it if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create a unique log filename with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Clear any existing logging handlers (prevents duplicate logs)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to both a file and the console
logging.basicConfig(
    level=logging.INFO,  # Log INFO and above (INFO, WARNING, ERROR, etc.)
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',  # Custom format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format for timestamps
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Write logs to the timestamped file
        logging.StreamHandler()  # Also show logs on console
    ]
)

# Note:
#Logger Setup (logging configuration)
# Purpose:
# Sets up a logging configuration that outputs logs to both a dynamically timestamped file 
# in a logs directory and the console.
#
# Key Points:
# - Creates logs directory if it doesn't exist.
# - Log filename includes date and time to avoid overwrites.
# - Removes existing root handlers to prevent duplicate logs.
# - Uses formatter including timestamp, line number, logger name, log level, and message.
