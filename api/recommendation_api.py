from flask import Flask, request, jsonify
import torch
from models.recommender import MultiModalRecommender
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import io
import logging
import os # For environment variables

app = Flask(__name__)

# Setup logging for production
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper() # Default to INFO, can be set via environment variable
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Load Model and Tokenizer/Transforms ---
MODEL_CHECKPOINT_PATH = os.environ.get('MODEL_CHECKPOINT_PATH', 'model_checkpoint.pth') # Path configurable via env var
USER_INPUT_DIM = int(os.environ.get('USER_INPUT_DIM', '10')) # User input dimension, configurable
USE_RULE_BASED = os.environ.get('USE_RULE_BASED', 'true').lower() == 'true' # Rule-based filtering toggle

try:
    model = MultiModalRecommender(user_input_dim=USER_INPUT_DIM, use_rule_based=USE_RULE_BASED)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    logger.info(f"Model loaded successfully from {MODEL_CHECKPOINT_PATH}. Rule-based filtering: {USE_RULE_BASED}")
except FileNotFoundError:
    logger.error(f"Model checkpoint file not found at {MODEL_CHECKPOINT_PATH}. Ensure the file exists and path is correct (env var MODEL_CHECKPOINT_PATH).")
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_CHECKPOINT_PATH}") # Re-raise to stop app if model load fails
except Exception as e:
    logger.error(f"Error loading model from {MODEL_CHECKPOINT_PATH}: {e}")
    raise Exception(f"Failed to load model: {e}") # Re-raise to stop app

try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    logger.info("Tokenizer and image transform initialized.")
except Exception as e:
    logger.error(f"Error initializing tokenizer or image transform: {e}")
    raise Exception(f"Failed to initialize tokenizer/transform: {e}") # Re-raise to stop app

# --- API Endpoint ---
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for content recommendation.
    Expects JSON or multipart/form-data with text, image, and user_features.
    Returns predicted rating as JSON.
    """
    logger.info("Recommendation request received.")
    try:
        # --- Input Parsing ---
        if request.content_type.startswith('multipart/form-data'):
            data = request.form
            image_file = request.files.get('image', None)
            if image_file is None:
                logger.warning("Multipart request received without image file.")
                return jsonify({'error': 'Image file is required in multipart form-data'}), 400
        else: # Assume JSON if not multipart
            data = request.get_json()
            image_file = None # Image needs to be provided in multipart if used, or base64 in JSON (not implemented here)
            if data is None:
                logger.warning("No JSON data received in request.")
                return jsonify({'error': 'Missing JSON input data'}), 400
            if 'image_path' not in data: # Expect image_path in JSON for this example, or handle base64 etc.
                logger.warning("JSON request without 'image_path'. Expecting image_path in JSON for this example.")
                return jsonify({'error': 'Image path is required in JSON request (image_path field)'}), 400
            image_path = data.get('image_path') # Get image path from JSON

        text = data.get('text', '')
        user_features_str = data.get('user_features', None) # Expect user_features as string

        # --- Text Preprocessing ---
        try:
            tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
        except Exception as e:
            logger.error(f"Error during text preprocessing: {e}")
            return jsonify({'error': 'Error processing text input'}), 400

        # --- Image Preprocessing ---
        try:
            if image_file: # Process image from uploaded file (multipart)
                image = Image.open(image_file.stream).convert('RGB')
            else: # Process image from path in JSON
                image = Image.open(image_path).convert('RGB') # Image.open handles path as string
            image_tensor = image_transform(image).unsqueeze(0).to(device) # Apply transform and add batch dimension
        except FileNotFoundError:
            logger.error(f"Image file not found at path: {image_path if not image_file else 'uploaded file'}")
            return jsonify({'error': 'Image file not found'}), 400
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': 'Error processing image file'}), 400

        # --- User Features Preprocessing ---
        try:
            if user_features_str:
                user_features = torch.tensor([float(x) for x in user_features_str.split(',')], dtype=torch.float).unsqueeze(0).to(device)
            else: # Default to zero user features if not provided
                user_features = torch.zeros((1, USER_INPUT_DIM)).to(device)
        except ValueError:
            logger.error(f"ValueError: Invalid user_features format: '{user_features_str}'. Expected comma-separated numbers.")
            return jsonify({'error': 'Invalid user_features format. Expecting comma-separated numbers in string.'}), 400
        except Exception as e:
            logger.error(f"Error processing user features: {e}")
            return jsonify({'error': 'Error processing user features'}), 400

        # --- Additional Info for Rule-based Filtering ---
        additional_info = {}
        if "is_new_user" in data:
            try:
                additional_info["is_new_user"] = data.get("is_new_user").lower() == 'true'
            except Exception:
                logger.warning("Could not parse 'is_new_user' value. Assuming False.")
                additional_info["is_new_user"] = False

        # --- Model Prediction ---
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                output = model(input_ids, attention_mask, image_tensor, user_features, additional_info)
            predicted_rating = output.item()
            logger.info(f"Recommendation generated successfully. Predicted rating: {predicted_rating:.4f}")
            return jsonify({'predicted_rating': predicted_rating})
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return jsonify({'error': 'Error generating recommendation'}), 500

    except Exception as top_e: # Catch any top-level exceptions
        logger.exception("Unhandled exception in recommendation endpoint: ") # Log full exception including traceback
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    # Run Flask app - in production, use a WSGI server like gunicorn
    port = int(os.environ.get('PORT', 5000)) # Port can be set via environment variable, default to 5000
    logger.info(f"Starting API server on port {port}, debug mode: False") # Debug mode is always False for production-like API
    app.run(host='0.0.0.0', port=port, debug=False)