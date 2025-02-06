import torch.nn as nn
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(TextEncoder, self).__init__()
        try:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            logger.info("BERT Model loaded for TextEncoder.")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise Exception(f"Failed to load BERT model: {e}")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        logger.info(f"TextEncoder initialized with output_dim={output_dim}")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for TextEncoder.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # Use [CLS] token for sentence representation
        features = self.fc(pooled)
        return features

if __name__ == '__main__':
    # Example usage
    import torch
    logging.basicConfig(level=logging.INFO)
    encoder = TextEncoder()
    # Dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128)) # Batch size 1, sequence length 128
    attention_mask = torch.ones((1, 128))
    try:
        features = encoder(input_ids, attention_mask)
        print("Text encoder output shape:", features.shape) # Expected: [1, 128]
    except Exception as e:
        print(f"Error in TextEncoder example: {e}")