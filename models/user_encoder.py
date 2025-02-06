import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class UserEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(UserEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        logger.info(f"UserEncoder initialized with input_dim={input_dim}, output_dim={output_dim}")

    def forward(self, user_features):
        """
        Forward pass for UserEncoder.
        """
        return self.fc(user_features)

if __name__ == '__main__':
    # Example usage
    import torch
    logging.basicConfig(level=logging.INFO)
    input_dim = 10 # Example user feature dimension
    encoder = UserEncoder(input_dim=input_dim)
    # Dummy input (batch of 1, user features of dimension input_dim)
    dummy_user_features = torch.randn(1, input_dim)
    try:
        features = encoder(dummy_user_features)
        print("User encoder output shape:", features.shape) # Expected: [1, 128]
    except Exception as e:
        print(f"Error in UserEncoder example: {e}")