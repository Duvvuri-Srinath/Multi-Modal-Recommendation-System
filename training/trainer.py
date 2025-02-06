import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def train_model(model, train_loader, num_epochs=10, lr=0.001, device='cuda', checkpoint_path='model_checkpoint.pth'):
    """
    Trains the model on a single GPU.
    Includes progress bar, logging, and model checkpoint saving.
    """
    model.to(device)
    criterion = nn.MSELoss() # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Starting training on {device} for {num_epochs} epochs.")
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as progress_bar:
            for batch in progress_bar:
                optimizer.zero_grad() # Clear gradients from previous batch

                # Move batch data to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image = batch['image'].to(device)
                user_features = batch['user_features'].to(device)
                rating = batch['rating'].to(device)
                additional_info = batch.get('additional_info', None) # Get additional info if available

                # Forward pass
                output = model(input_ids, attention_mask, image, user_features, additional_info)
                loss = criterion(output.squeeze(), rating.float()) # Calculate loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)}) # Update progress bar with current loss

        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

    logger.info("Training complete.")
    # Save model checkpoint after training
    try:
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving model checkpoint: {e}")

if __name__ == '__main__':
    # Example usage (requires dataset and model to be defined - using dummies for demonstration)
    import torch.utils.data as data
    from models.recommender import MultiModalRecommender
    logging.basicConfig(level=logging.INFO)

    # Dummy Dataset for testing
    class DummyDataset(data.Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx):
            return {'input_ids': torch.randint(0, 100, (128,)),
                    'attention_mask': torch.ones(128),
                    'image': torch.randn(3, 224, 224),
                    'user_features': torch.randn(10),
                    'rating': torch.tensor(3.5),
                    'additional_info': {'is_new_user': False}}
    dummy_dataset = DummyDataset()
    dummy_loader = data.DataLoader(dummy_dataset, batch_size=16)

    # Dummy Model
    dummy_model = MultiModalRecommender(user_input_dim=10, use_rule_based=False)

    try:
        train_model(dummy_model, dummy_loader, num_epochs=2, device='cpu', checkpoint_path='dummy_model_checkpoint.pth')
        print("Dummy training run completed (see logs for details). Check for 'dummy_model_checkpoint.pth' file.")
    except Exception as e:
        print(f"Error in example training run: {e}")