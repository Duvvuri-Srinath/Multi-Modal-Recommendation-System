import argparse
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler # Import DistributedSampler for distributed training
import numpy as np
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
import pandas as pd # Import pandas for data handling
import logging
import os # For environment variables

# --- Custom Modules Import ---
from data.data_loader import load_data, get_train_test_split
from training.trainer import train_model
from training.distributed_trainer import train_model_distributed
from training.evaluation import evaluate_model
from models.recommender import MultiModalRecommender

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """
    Dataset class for multi-modal movie data.
    Loads data from DataFrame and preprocesses text, image, and user features.
    """
    def __init__(self, df, tokenizer, image_transform):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        logger.info("MultiModalDataset initialized.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Text Processing
            tokens = self.tokenizer.encode_plus(row['summary'], add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)

            # Image Processing
            image = Image.open(row['image_path']).convert('RGB')
            image = self.image_transform(image)

            # User Features Processing
            user_features = np.array([float(x) for x in row['user_features'].split(',')])
            user_features = torch.tensor(user_features, dtype=torch.float)

            # Rating
            rating = torch.tensor(row['rating'], dtype=torch.float)

            # Additional Info (e.g., new user flag)
            additional_info = {"is_new_user": row.get('is_new_user', False)}

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image': image,
                'user_features': user_features,
                'rating': rating,
                'additional_info': additional_info
            }
        except FileNotFoundError as fnf_e:
            logger.error(f"FileNotFoundError in dataset __getitem__ at index {idx}: {fnf_e}")
            raise # Re-raise to be handled by DataLoader or outer scope
        except Exception as e:
            logger.error(f"Error processing data at index {idx}: {e}")
            raise # Re-raise to be handled

def main(args):
    """
    Main function to run training, evaluation, and model saving.
    Supports both single-node and distributed training based on arguments.
    """
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate # Get learning rate from arguments
    use_distributed = args.distributed
    checkpoint_path = args.checkpoint_path # Get checkpoint path from arguments
    data_csv_path = args.data_path # Get data path from arguments

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Running in distributed mode: {use_distributed}")

    # --- Data Loading and Preprocessing ---
    try:
        data_df = load_data(data_csv_path) # Load data using data_loader module
        train_df, test_df = get_train_test_split(data_df) # Split into train/test
    except Exception as e:
        logger.error(f"Data loading or splitting error: {e}")
        return

    # --- Tokenizer and Image Transform Initialization ---
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        image_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    except Exception as e:
        logger.error(f"Tokenizer or image transform initialization error: {e}")
        return

    # --- Dataset and DataLoader Creation ---
    try:
        train_dataset = MultiModalDataset(train_df, tokenizer, image_transform)
        test_dataset = MultiModalDataset(test_df, tokenizer, image_transform)

        if use_distributed:
            train_sampler = DistributedSampler(train_dataset) # Create DistributedSampler for training dataset
            shuffle_train = False # Shuffling is handled by DistributedSampler
        else:
            train_sampler = None
            shuffle_train = True

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler, num_workers=args.num_workers) # Add num_workers
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers) # num_workers for test loader as well
    except Exception as e:
        logger.error(f"Dataset or DataLoader creation error: {e}")
        return

    # --- Model Initialization ---
    try:
        model = MultiModalRecommender(user_input_dim=10, use_rule_based=True) # Initialize model
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        return

    # --- Training ---
    try:
        if use_distributed:
            train_model_distributed(model, train_loader, num_epochs=num_epochs, lr=learning_rate, checkpoint_path=checkpoint_path)
        else:
            train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate, device=device, checkpoint_path=checkpoint_path)
    except Exception as e:
        logger.error(f"Training error: {e}")
        return

    # --- Evaluation ---
    try:
        evaluate_model(model, test_loader, device=device)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return

    # --- Model Saving (already in train_model/train_model_distributed, but can save again explicitly if needed) ---
    logger.info(f"Model checkpoint saved to {checkpoint_path} (path from arguments).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Modal Recommendation Training and Evaluation Script")

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training (using torch.distributed.launch)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading') # Add num_workers argument

    # --- Path and Checkpoint Arguments ---
    parser.add_argument('--data_path', type=str, default='data/movies.csv', help='Path to the movies CSV data file') # Data path argument
    parser.add_argument('--checkpoint_path', type=str, default='model_checkpoint.pth', help='Path to save the model checkpoint') # Checkpoint path argument

    args = parser.parse_args()

    # --- Environment Variable Configuration Example ---
    # You could also load configurations from environment variables here if needed.
    # For example, to override data_path from environment:
    # data_path_env = os.environ.get('DATA_PATH')
    # if data_path_env:
    #     args.data_path = data_path_env
    # logger.info(f"Data path from environment variable: {args.data_path}")

    main(args)