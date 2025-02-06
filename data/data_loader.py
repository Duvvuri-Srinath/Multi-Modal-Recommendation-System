import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__) # Set up logger for this module

def load_data(csv_path):
    """
    Loads data from a CSV file.
    Handles potential file not found errors.
    """
    try:
        data = pd.read_csv(csv_path)
        logger.info(f"Data loaded successfully from {csv_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found at {csv_path}")
        raise FileNotFoundError(f"Data CSV file not found at: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading data from {csv_path}: {e}")
        raise Exception(f"Error loading data: {e}")

def get_train_test_split(data, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.
    Uses stratified splitting if labels are provided (can be extended).
    """
    try:
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        logger.info(f"Data split into train/test sets ({1-test_size:.2f}/{test_size:.2f} split)")
        return train, test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise Exception(f"Error splitting data: {e}")

if __name__ == '__main__':
    # Example usage and basic test
    logging.basicConfig(level=logging.INFO) # Setup basic logging for example run
    try:
        data = load_data('data/movies.csv') # Assuming movies.csv is in the data folder
        print("Data loaded successfully. First 5 rows:")
        print(data.head())
        train_data, test_data = get_train_test_split(data)
        print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")
    except FileNotFoundError:
        print("Error: movies.csv not found. Make sure it's in the data/ directory or provide correct path.")
    except Exception as e:
        print(f"An error occurred: {e}")