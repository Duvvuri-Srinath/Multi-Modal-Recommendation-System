from transformers import BertTokenizer
import logging

logger = logging.getLogger(__name__)

# Initialize tokenizer once globally for efficiency
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info("BERT Tokenizer initialized.")
except Exception as e:
    logger.error(f"Error initializing BERT Tokenizer: {e}")
    raise Exception(f"Failed to initialize BERT Tokenizer: {e}")

def preprocess_text(text, max_length=128):
    """
    Preprocesses text using BERT tokenizer.
    Handles potential errors during encoding.
    """
    try:
        tokens = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise Exception(f"Text preprocessing failed: {e}")

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    example_text = "This is an example movie summary."
    try:
        input_ids, attention_mask = preprocess_text(example_text)
        print("Preprocessed input_ids shape:", input_ids.shape)
        print("Preprocessed attention_mask shape:", attention_mask.shape)
    except Exception as e:
        print(f"Error in example usage: {e}")