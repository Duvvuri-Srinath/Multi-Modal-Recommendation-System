import logging

logger = logging.getLogger(__name__)

def apply_rule_based_adjustment(prediction, user_features, additional_info):
    """
    Apply rule-based adjustments to the prediction.
    Example rule: if user is 'new', reduce predicted rating.
    Can be extended with more complex rules based on user_features and additional_info.
    """
    adjusted_prediction = prediction # Initialize with original prediction

    if additional_info and additional_info.get("is_new_user", False):
        adjusted_prediction = prediction * 0.9 # Reduce rating by 10% for new users
        logger.info("Rule-based adjustment applied: -10% for new user.")
    # Add more rules here as needed, e.g., based on user_features thresholds, content categories etc.

    return adjusted_prediction

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    initial_prediction = 4.5 # Example predicted rating
    dummy_user_features = [0.1, 0.2, 0.3] # Not used in this simple rule, but could be used for more complex rules
    additional_info_new_user = {"is_new_user": True}
    additional_info_existing_user = {"is_new_user": False}

    adjusted_rating_new = apply_rule_based_adjustment(initial_prediction, dummy_user_features, additional_info_new_user)
    adjusted_rating_existing = apply_rule_based_adjustment(initial_prediction, dummy_user_features, additional_info_existing_user)

    print(f"Initial prediction: {initial_prediction}")
    print(f"Adjusted prediction (new user): {adjusted_rating_new}") # Expected: 4.05
    print(f"Adjusted prediction (existing user): {adjusted_rating_existing}") # Expected: 4.5