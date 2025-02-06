import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluates the model on the test dataset.
    Calculates and logs Root Mean Squared Error (RMSE).
    """
    model.eval() # Set model to evaluation mode
    model.to(device)
    predictions = []
    ground_truths = []

    logger.info("Starting model evaluation.")
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            user_features = batch['user_features'].to(device)
            rating = batch['rating'].to(device)
            additional_info = batch.get('additional_info', None)

            output = model(input_ids, attention_mask, image, user_features, additional_info)
            predictions.extend(output.squeeze().cpu().numpy()) # Move predictions to CPU and store as numpy array
            ground_truths.extend(rating.cpu().numpy()) # Move ground truth ratings to CPU and store

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    logger.info(f"Evaluation complete. Test RMSE: {rmse:.4f}")
    print(f"Test RMSE: {rmse:.4f}") # Also print to console for immediate feedback
    return rmse

if __name__ == '__main__':
    # Example usage (requires test dataset and trained model - using dummies)
    import torch.utils.data as data
    from models.recommender import MultiModalRecommender
    logging.basicConfig(level=logging.INFO)

    # Dummy Test Dataset
    class DummyDataset(data.Dataset):
        def __len__(self): return 50
        def __getitem__(self, idx):
            return {'input_ids': torch.randint(0, 100, (128,)),
                    'attention_mask': torch.ones(128),
                    'image': torch.randn(3, 224, 224),
                    'user_features': torch.randn(10),
                    'rating': torch.tensor(np.random.uniform(1, 5)), # Random ratings between 1 and 5
                    'additional_info': {'is_new_user': False}}
    dummy_test_dataset = DummyDataset()
    dummy_test_loader = data.DataLoader(dummy_test_dataset, batch_size=16)

    # Dummy Model (load a pre-saved dummy checkpoint if you have one, or initialize a new one)
    dummy_model = MultiModalRecommender(user_input_dim=10, use_rule_based=False)
    # For testing evaluation, you can load a dummy checkpoint if you have saved one from dummy training, e.g.,
    # dummy_model.load_state_dict(torch.load('dummy_model_checkpoint.pth', map_location='cpu')) # Load to CPU for example
    # Or just run evaluation on a randomly initialized model:

    try:
        rmse_score = evaluate_model(dummy_model, dummy_test_loader, device='cpu')
        print(f"Dummy evaluation run completed. RMSE: {rmse_score:.4f} (see logs for details)")
    except Exception as e:
        print(f"Error in example evaluation run: {e}")