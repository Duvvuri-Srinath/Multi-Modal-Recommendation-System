import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_user_behavior(features_str):
    """
    Preprocesses user behavior features from a comma-separated string.
    Handles cases with invalid input strings.
    """
    try:
        features = np.array([float(x) for x in features_str.split(',')])
        return features
    except ValueError: # Catch error if string cannot be converted to float
        logger.error(f"ValueError: Invalid user feature string format: '{features_str}'. Expected comma-separated numbers.")
        raise ValueError(f"Invalid user feature format. Expected comma-separated numbers in string: '{features_str}'")
    except Exception as e:
        logger.error(f"Error preprocessing user behavior features: {e}")
        raise Exception(f"Error preprocessing user features: {e}")

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    valid_features_str = "1.0,2.5,3.0,0.5,1.2,0.8,2.0,1.5,0.9,1.1"
    invalid_features_str = "1.0,invalid,3.0"
    try:
        processed_features = preprocess_user_behavior(valid_features_str)
        print("Processed user features:", processed_features)
    except ValueError as ve:
        print(f"ValueError in example usage (valid input): {ve}")
    except Exception as e:
        print(f"Error in example usage (valid input): {e}")

    try:
        preprocess_user_behavior(invalid_features_str) # This should raise a ValueError
    except ValueError as ve:
        print(f"ValueError caught correctly for invalid input: {ve}")
    except Exception as e:
        print(f"Error in example usage (invalid input - expected ValueError, got): {e}")