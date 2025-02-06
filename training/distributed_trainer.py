import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def init_distributed(backend="nccl"):
    """
    Initializes distributed training environment using PyTorch Distributed.
    Sets up process group and device ID for current process.
    """
    dist.init_process_group(backend=backend) # Initialize process group, using NCCL backend (optimized for NVIDIA GPUs)
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # Get local rank from environment variable set by launcher
    torch.cuda.set_device(local_rank) # Set current device to local rank (GPU assignment for this process)
    logger.info(f"Initialized distributed training. Rank: {dist.get_rank()}, Local Rank: {local_rank}, World Size: {dist.get_world_size()}")
    return local_rank

def train_model_distributed(model, train_loader, num_epochs=10, lr=0.001, checkpoint_path='distributed_model_checkpoint.pth'):
    """
    Trains the model using distributed data parallel (DDP) across multiple GPUs/nodes.
    Only rank 0 process logs training progress and saves the model checkpoint.
    """
    local_rank = init_distributed() # Initialize distributed environment
    device = torch.device(f"cuda:{local_rank}") # Device for this process
    model.to(device) # Move model to current device

    # Wrap model with DDP - DistributedDataParallel
    model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_ddp.parameters(), lr=lr)

    logger.info(f"Starting distributed training on rank {dist.get_rank()} for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        running_loss = 0.0
        # Use DistributedSampler to ensure each process gets a unique subset of data
        train_sampler = train_loader.sampler
        if isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
            train_sampler.set_epoch(epoch) # Important for shuffling in distributed training

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Rank {dist.get_rank()}]", disable=(dist.get_rank() != 0)) as progress_bar: # Only show progress bar for rank 0
            for batch in progress_bar:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image = batch['image'].to(device)
                user_features = batch['user_features'].to(device)
                rating = batch['rating'].to(device)
                additional_info = batch.get('additional_info', None)

                output = model_ddp(input_ids, attention_mask, image, user_features, additional_info) # Use model_ddp for forward pass
                loss = criterion(output.squeeze(), rating.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if dist.get_rank() == 0: # Only update progress bar for rank 0
                    progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

        epoch_loss = running_loss / len(train_loader)
        if dist.get_rank() == 0: # Only log epoch loss from rank 0 to avoid duplicate logs
            logger.info(f"Epoch {epoch+1}/{num_epochs} Loss (Rank 0): {epoch_loss:.4f}")

    if dist.get_rank() == 0: # Only rank 0 process saves model and prints completion message
        logger.info("Distributed training complete.")
        try:
            # Save only the model's state_dict, not the DDP wrapper
            torch.save(model.module.state_dict(), checkpoint_path) # Access original model via .module attribute
            logger.info(f"Distributed model checkpoint saved to {checkpoint_path} from rank 0.")
        except Exception as e:
            logger.error(f"Error saving distributed model checkpoint from rank 0: {e}")
    dist.destroy_process_group() # Clean up distributed process group

if __name__ == '__main__':
    # Example Usage (similar dummy setup as in trainer.py, but for distributed)
    import torch.utils.data as data
    from models.recommender import MultiModalRecommender
    logging.basicConfig(level=logging.INFO)

    # Dummy Dataset and DataLoader (using DistributedSampler for distributed training)
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
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(dummy_dataset, num_replicas=2, rank=0) # Example for 2 processes
    dummy_loader = data.DataLoader(dummy_dataset, batch_size=16, sampler=distributed_sampler)

    # Dummy Model
    dummy_model = MultiModalRecommender(user_input_dim=10, use_rule_based=False)

    # To run this example in distributed mode, you would typically use torch.distributed.launch or similar.
    # Example command for 2 GPUs:
    # python -m torch.distributed.launch --nproc_per_node=2 distributed_trainer.py

    try:
        # In a real distributed setup, this script would be launched by torch.distributed.launch
        # For local testing without actual distributed setup, this will run in 'single-process' mode and may error out due to dist.init_process_group.
        # Comment out init_distributed and DDP wrapping for single process testing if needed.
        train_model_distributed(dummy_model, dummy_loader, num_epochs=1, checkpoint_path='dummy_distributed_model_checkpoint.pth')
        print("Dummy distributed training run initiated (see logs for details). Check for 'dummy_distributed_model_checkpoint.pth' from rank 0.")
    except Exception as e:
        print(f"Error in example distributed training run: {e}")