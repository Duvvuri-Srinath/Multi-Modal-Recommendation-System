import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1): # Output_dim=1 for regression (rating)
        super(FusionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        logger.info(f"FusionModel initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")

    def forward(self, concatenated_features):
        """
        Forward pass for FusionModel.
        """
        return self.fc(concatenated_features)

if __name__ == '__main__':
    # Example usage
    import torch
    logging.basicConfig(level=logging.INFO)
    input_dim = 128 * 3 # Assuming embeddings from text, image, user encoders are 128 each
    fusion_model = FusionModel(input_dim=input_dim)
    # Dummy input (batch of 1, concatenated features)
    dummy_features = torch.randn(1, input_dim)
    try:
        output = fusion_model(dummy_features)
        print("Fusion model output shape:", output.shape) # Expected: [1, 1] for rating prediction
    except Exception as e:
        print(f"Error in FusionModel example: {e}")