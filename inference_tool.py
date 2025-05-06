import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Layer
# Import the specific preprocess_input function used by the base model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import json
import os
from PIL import Image # Using Pillow for image loading



# 1. Define the custom layer used during training
# This definition must match the one used when the model was saved
class PreprocessInputLayer(Layer):
    def __init__(self, name="preprocess_input", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        # Apply the same preprocessing as used with the base model (ResNet50 in this case)
        return preprocess_input(inputs)

    def get_config(self):
        # This is important for Keras to be able to save/load the layer
        base_config = super().get_config()
        # Add any custom parameters here if you had them
        return base_config

# --- Configuration ---
# Define the paths to your saved model files
MODEL_CONFIG_PATH = 'config.json'
MODEL_WEIGHTS_PATH = 'model.weights.h5'
METADATA_PATH = 'metadata.json' # Assuming metadata.json exists and contains class names

# Define the input image size used during training (from load_and_visualize_dataset)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Model Loading Function ---
def load_inference_model(config_path, weights_path):
    """Loads the model architecture from JSON and then loads the weights."""
    print(f"Loading model architecture from {config_path}...")
    # Load the model architecture from the config.json file
    with open(config_path, 'r') as f:
        model_json = f.read()

    # When loading the model architecture, provide the custom object(s)
    # Keras needs to know how to recreate your custom layer(s)
    try:
        model = model_from_json(model_json, custom_objects={'PreprocessInputLayer': PreprocessInputLayer})
        print("Model architecture loaded successfully.")
    except Exception as e:
        print(f"Error loading model architecture: {e}")
        print("Ensure your config.json is valid and the custom_objects dictionary includes all necessary custom layers.")
        raise

    print(f"Loading model weights from {weights_path}...")
    # Load the weights into the model
    try:
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure your model.weights.h5 file is in the correct path and matches the loaded architecture.")
        raise

    # Optional: You can compile the model for potentially faster inference,
    # but it's not strictly necessary just for prediction (`model.predict()`).
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    return model

# --- Metadata Loading Function ---
def load_class_names(metadata_path):
    """Loads the class names from the metadata.json file."""
    print(f"Loading class names from {metadata_path}...")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Assuming metadata.json has a structure like {"class_names": ["Class1", "Class2", ...]}
        # Adjust 'class_names' key if your metadata.json uses a different key
        class_names = metadata.get('class_names')

        if not class_names:
             # Fallback or error handling if the key is not found
             # Based on your organize_images function, these were the classes:
             print("Warning: 'class_names' key not found in metadata.json.")
             # Use the classes from organize_images as a fallback, respecting the '/' -> '_' replacement
             fallback_classes = ['Church', 'Enough_Satisfied', 'Friend', 'Love', 'Me', 'Mosque', 'Seat', 'Temple', 'You']
             print(f"Using hardcoded fallback class names: {fallback_classes}")
             class_names = fallback_classes
             # IMPORTANT: Ensure this fallback list is in the *exact same order* as your model's output classes.
             # The best approach is to ensure metadata.json is correct.

        print("Class names loaded:", class_names)
        return class_names

    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please ensure metadata.json exists and contains the class names.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metadata_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading metadata: {e}")
        raise


# --- Image Preprocessing Function ---
def preprocess_image(image_input, img_height, img_width):
    """
    Loads and preprocesses a single image for inference.
    Args:
        image_input: Path to the image file (str) or a PIL Image object.
        img_height: The required image height.
        img_width: The required image width.
    Returns:
        A NumPy array representing the preprocessed image with a batch dimension.
    """
    if isinstance(image_input, str): # Input is a file path
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found at {image_input}")
        # Load the image using PIL
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image): # Input is a PIL Image object
        img = image_input
    else:
        raise TypeError("image_input must be a file path (str) or PIL Image object.")

    # Ensure image is in RGB format (handle grayscale or other formats)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image to the target size
    img = img.resize((img_width, img_height))

    # Convert the PIL image to a NumPy array
    img_array = np.array(img)

    # Add a batch dimension at the beginning (models expect input in batches)
    # The shape goes from (height, width, channels) to (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: We *do not* manually apply `preprocess_input` here.
    # The `PreprocessInputLayer` is part of the loaded model architecture,
    # and the model will apply it automatically when you call `model.predict()`.
    # The data augmentation layers (RandomFlip, etc.) are also part of the model
    # but are typically inactive during inference.

    return img_array


# --- Prediction Function ---
def predict_sign(model, image_input, class_names):
    """
    Performs inference on a single image.
    Args:
        model: The loaded Keras model.
        image_input: Path to the image file (str) or a PIL Image object.
        class_names: A list of class names in the order corresponding to model output.
    Returns:
        A tuple containing:
        - predicted_class_name (str): The name of the predicted sign.
        - confidence (float): The confidence score for the predicted class.
        - all_predictions (np.array): The raw probability outputs for all classes.
    """
    # Preprocess the image
    img_tensor = preprocess_image(image_input, IMG_HEIGHT, IMG_WIDTH)

    # Make prediction using the model
    # predictions will be a list/array of shape (1, num_classes) because of the batch dimension
    predictions = model.predict(img_tensor)

    # Get the probabilities for the single image in the batch
    probabilities = predictions[0]

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(probabilities)

    # Get the predicted class name and its confidence score
    predicted_class_name = class_names[predicted_class_index]
    confidence = probabilities[predicted_class_index]

    return predicted_class_name, confidence, probabilities

# --- Example Usage (for testing the script directly) ---
if __name__ == "__main__":
    # Check if model files exist
    if not all(os.path.exists(p) for p in [MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH, METADATA_PATH]):
        print("Error: Model files (config.json, model.weights.h5, metadata.json) not found.")
        print("Please place these files in the same directory as this script.")
        # Optionally, create dummy files for demonstration if you don't have the real ones yet
        # WARNING: Dummy files will not work for actual prediction!
        print("\nAttempting to create dummy placeholder files...")
        try:
            # Dummy config.json (simplified structure - replace with your actual content)
            dummy_config_content = {
              "class_name": "Sequential",
              "config": {
                "name": "sequential",
                "layers": [
                  {"class_name": "InputLayer", "config": {"batch_input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3], "dtype": "float32", "sparse": False, "ragged": False, "name": "input_1"}},
                  # Include your custom layer and other relevant layers from your actual config
                   {"class_name": "__main__.PreprocessInputLayer", "config": {"name": "preprocess_input", "dtype": "float32"}},
                   # Add placeholders for ResNet50 and dense layers - structure must match
                   # This dummy config WILL LIKELY FAIL to load your actual weights.
                   # Replace with the real config.json content!
                   {"class_name": "Dense", "config": {"name": "dummy_output", "units": 9, "activation": "softmax"}} # Minimal output layer placeholder
                ]
              }
            }
            with open(MODEL_CONFIG_PATH, 'w') as f:
                 json.dump(dummy_config_content, f, indent=4)
            print(f"Created dummy {MODEL_CONFIG_PATH}. Replace with your actual file.")

            # Dummy weights file (empty - replace with your actual weights)
            # This is just to prevent load_inference_model from failing immediately on file existence check
            import h5py
            with h5py.File(MODEL_WEIGHTS_PATH, 'w') as f:
                pass # Create an empty HDF5 file
            print(f"Created dummy {MODEL_WEIGHTS_PATH}. Replace with your actual file.")


            # Dummy metadata.json
            dummy_metadata_content = {"class_names": ['Church', 'Enough_Satisfied', 'Friend', 'Love', 'Me', 'Mosque', 'Seat', 'Temple', 'You']}
            with open(METADATA_PATH, 'w') as f:
                 json.dump(dummy_metadata_content, f, indent=4)
            print(f"Created dummy {METADATA_PATH}. Replace with your actual file.")

        except Exception as e:
            print(f"Failed to create dummy files: {e}")
            print("Cannot proceed without model files.")
            exit()

        print("\nIMPORTANT: The dummy files are placeholders. You MUST replace them with your actual model files!")
        print("Attempting to load the dummy files (this will likely fail during weight loading)...")


    # Load the model and class names once when the script starts
    try:
        model = load_inference_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
        class_names = load_class_names(METADATA_PATH)
    except Exception as e:
        print(f"Fatal error during model/metadata loading: {e}")
        print("Exiting.")
        exit()

    # --- How to use the prediction function ---
    # Replace 'path/to/your/test_image.jpg' with the actual path to an image you want to classify
    # For demonstration, let's create a dummy blank image if no test image path is provided
    test_image_path = 'ImageID_0AB8B2QN.jpg' # Name for a sample image

    if not os.path.exists(test_image_path):
        print(f"\nCreating a blank sample image at {test_image_path} for testing...")
        try:
            # Create a simple red blank image
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (255, 0, 0))
            img.save(test_image_path)
            print("Sample image created. Replace this with a real sign image for a meaningful prediction.")
        except Exception as e:
            print(f"Could not create sample image: {e}")
            test_image_path = None # Indicate that no test image is available


    if test_image_path and os.path.exists(test_image_path):
        print(f"\nPerforming inference on {test_image_path}...")
        try:
            predicted_class, confidence, all_predictions = predict_sign(model, test_image_path, class_names)

            print("\n--- Inference Results ---")
            print(f"Predicted Sign: **{predicted_class}**")
            print(f"Confidence: {confidence:.2f}") # Format confidence to 2 decimal places

            # Optional: Print probabilities for all classes
            print("\nAll Class Probabilities:")
            for i, prob in enumerate(all_predictions):
                 print(f"  {class_names[i]}: {prob:.4f}")

        except FileNotFoundError as e:
            print(f"Error during prediction: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
    else:
        print("\nNo test image available to perform inference.")