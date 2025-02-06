from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

# Initialize image transform once globally for efficiency
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
logger.info("Image transformation pipeline initialized.")

def preprocess_image(image_path):
    """
    Preprocesses an image from a given path.
    Handles file not found and image opening errors.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image_transform(image)
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e: # Catches PIL related errors like corrupted images
        logger.error(f"Error processing image at {image_path}: {e}")
        raise Exception(f"Image processing failed for {image_path}: {e}")

if __name__ == '__main__':
    # Example usage (assuming you have an image file 'test_image.jpg' in data/)
    logging.basicConfig(level=logging.INFO)
    try:
        processed_image = preprocess_image('data/test_image.jpg') # Replace with a valid image path for testing
        print("Preprocessed image shape:", processed_image.shape)
    except FileNotFoundError:
        print("Error: test_image.jpg not found in data/. Place a test image there or change path.")
    except Exception as e:
        print(f"Error in example usage: {e}")