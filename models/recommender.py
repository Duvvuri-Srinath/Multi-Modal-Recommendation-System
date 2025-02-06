import torch
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.user_encoder import UserEncoder
from models.fusion_model import FusionModel
from models.rule_based_filter import apply_rule_based_adjustment
import logging

logger = logging.getLogger(__name__)

class MultiModalRecommender(nn.Module):
    def __init__(self, user_input_dim, fusion_hidden_dim=256, embedding_dim=128, use_rule_based=False):
        super(MultiModalRecommender, self).__init__()
        self.text_encoder = TextEncoder(output_dim=embedding_dim)
        self.image_encoder = ImageEncoder(output_dim=embedding_dim)
        self.user_encoder = UserEncoder(input_dim=user_input_dim, output_dim=embedding_dim)
        # Fuse three modalities: text, image, and user features
        self.fusion = FusionModel(input_dim=embedding_dim * 3, hidden_dim=fusion_hidden_dim, output_dim=1) # Regression task (rating)
        self.use_rule_based = use_rule_based
        logger.info(f"MultiModalRecommender initialized. Rule-based filtering: {use_rule_based}")

    def forward(self, input_ids, attention_mask, image, user_features, additional_info=None):
        """
        Forward pass for MultiModalRecommender.
        Combines outputs from text, image, and user encoders, fuses them,
        and applies rule-based filtering if enabled.
        """
        text_feat = self.text_encoder(input_ids, attention_mask)
        image_feat = self.image_encoder(image)
        user_feat = self.user_encoder(user_features)

        # Concatenate features from all modalities
        fused_features = torch.cat([text_feat, image_feat, user_feat], dim=1)
        prediction = self.fusion(fused_features)

        # Apply rule-based adjustments if enabled
        if self.use_rule_based:
            prediction = apply_rule_based_adjustment(prediction, user_features, additional_info)

        return prediction

if __name__ == '__main__':
    # Example usage
    import torch
    logging.basicConfig(level=logging.INFO)
    user_input_dim = 10 # Example user feature dimension
    model = MultiModalRecommender(user_input_dim=user_input_dim, use_rule_based=True)

    # Dummy inputs
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128))
    attention_mask = torch.ones((1, 128))
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_user_features = torch.randn(1, user_input_dim)
    additional_info = {"is_new_user": True}

    try:
        output = model(input_ids, attention_mask, dummy_image, dummy_user_features, additional_info)
        print("Recommender output (predicted rating):", output.item())
    except Exception as e:
        print(f"Error in MultiModalRecommender example: {e}")