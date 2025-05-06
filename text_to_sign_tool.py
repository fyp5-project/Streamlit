# text_to_sign_tool.py
import pandas as pd
import json
import os
import cv2
import numpy as np
import tempfile # To create temporary files/directories
import shutil   # To clean up temporary directories

# --- Configuration (Relative Paths) ---
# Define paths relative to the project root or data directory
# Assuming 'data' folder is one level up from this script if placed in a subdir,
# but simpler if data is in a sibling folder to the scripts. Let's assume sibling folder 'data'.
DATA_DIR = 'data'
LABELS_CSV = os.path.join(DATA_DIR, 'labels.csv')
TEXT_TO_CLASS_JSON = os.path.join(DATA_DIR, 'text_to_class.json')
# The actual video files should be located according to the paths in labels.csv,
# relative to the location of labels.csv

# --- Global Data (Loaded once) ---
# Use variables to hold loaded dataframes and mappings
labels_df = None
text_to_class_mapping = None

# --- Data Loading Functions ---
# text_to_sign_tool.py (Updated load_labels_df function)
import pandas as pd
import json
import os
import cv2
import numpy as np
import tempfile
import shutil
import speech_recognition as sr # Keep this import if you plan to enable audio later

# --- Configuration (Relative Paths) ---
DATA_DIR = 'data'
LABELS_CSV = os.path.join(DATA_DIR, 'labels.csv')
TEXT_TO_CLASS_JSON = os.path.join(DATA_DIR, 'text_to_class.json')

# --- Global Data (Loaded once) ---
labels_df = None
text_to_class_mapping = None

# --- Data Loading Functions --

# --- Data Loading Functions ---
def load_labels_df(labels_csv_path):
    """
    Loads the video labels CSV, handling potential encoding issues,
    validates columns, and saves the corrected DataFrame back to the source file.
    Ensures necessary columns exist and paths are handled.
    """
    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Labels CSV not found at {labels_csv_path}")

    df = None
    # Attempt reading with common encodings, handling errors during decode
    # Start with UTF-8 as it's standard, then try others like utf-7
    encodings_to_try = ['utf-8', 'utf-7', 'latin-1']

    print(f"Attempting to read labels CSV from {labels_csv_path}...")
    for encoding in encodings_to_try:
        try:
            # Use encoding_errors='replace' which is the correct parameter for handling
            # characters that cannot be decoded with the specified encoding.
            # Requires pandas >= 1.3.0
            df = pd.read_csv(labels_csv_path, encoding=encoding, encoding_errors='replace')
            print(f"Successfully read labels CSV with {encoding} encoding.")
            # Assuming if it reads, this encoding is acceptable for now
            break # Exit loop if reading is successful
        except TypeError:
             # This catch is for older pandas versions that don't support encoding_errors
             print(f"Warning: Your pandas version does not support 'encoding_errors'. Trying read with {encoding} without it (might fail on bad characters).")
             try:
                 # Attempt reading without explicit error handling, rely on default
                 df = pd.read_csv(labels_csv_path, encoding=encoding)
                 print(f"Successfully read labels CSV with {encoding} encoding (without error handling).")
                 break # Exit loop if reading is successful
             except Exception as e_no_errors:
                 print(f"Reading with {encoding} (no error handling) failed: {e_no_errors}")
        except Exception as e:
            # Catch other reading errors (e.g., FileNotFoundError, parser errors)
            print(f"Reading with {encoding} encoding failed: {e}")

    if df is None:
         # If df is still None after trying all encodings
         raise IOError(f"Failed to read labels CSV from {labels_csv_path} with any attempted encoding ({', '.join(encodings_to_try)}).")


    # --- Validation checks (Keep these!) ---
    # Check for required columns *after* reading
    required_columns = ['file_path', 'class']
    if not all(col in df.columns for col in required_columns):
         # If the expected columns are not found, raise an error
         found_columns = df.columns.tolist()
         raise ValueError(f"Labels CSV must contain {required_columns} columns. Found: {found_columns}")


    # Drop rows where required columns might be missing after reading/replacing errors
    original_rows = len(df)
    df.dropna(subset=required_columns, inplace=True)
    if len(df) < original_rows:
        print(f"Warning: Dropped {original_rows - len(df)} rows with missing values in {required_columns} after reading.")

    if df.empty:
        raise ValueError(f"Labels CSV from {labels_csv_path} is empty or contains no valid entries after cleaning.")

    # --- Save the Cleaned DataFrame Back to the File ---
    # WARNING: This modifies the source file within the loading function!
    # This is NOT recommended for general Streamlit caching practices.
    print(f"Saving corrected DataFrame back to {labels_csv_path} (UTF-8 encoding)...")
    try:
        df.to_csv(labels_csv_path, encoding='utf-8', index=False)
        print("Successfully saved the corrected labels CSV.")
    except Exception as e:
        print(f"Warning: Failed to save the corrected labels CSV back to {labels_csv_path}: {e}")
        # Decide if this should be a critical error or just a warning.
        # For now, we'll let the app proceed with the DataFrame in memory,
        # but the file on disk might not be updated.


    # --- Path Handling (Ensure paths are correct relative to DATA_DIR) ---
    # Check if the file_path column entries look like relative paths or need correction
    # This logic depends heavily on your specific CSV content
    # Example: If paths are just filenames, prepend 'sign_videos/'
    # If paths are still Google Drive paths, you need to replace the prefix.
    # A robust script would handle this *before* running the app.
    # For now, let's assume the paths in the CSV are correctly relative to DATA_DIR like 'sign_videos/...'

    # You might want to add a check here to ensure paths don't start with '/' (absolute)
    # or contain problematic prefixes like 'C:\' or '/content/drive/'

    # --- Optional Video File Existence Validation ---
    # Add the optional validation check here if you want to ensure videos exist.
    # (Code is commented out in the previous response)
    # Be aware this adds load time.

    print(f"Loaded {len(df)} valid video entries from {labels_csv_path}.")

    return df


def load_text_to_class_mapping(json_path):
    """Loads or creates the text-to-class mapping."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            mapping = json.load(f)
        print(f"Loaded text_to_class mapping from {json_path}")
    else:
        # Define the text-to-class mapping (case-insensitive keys)
        mapping = {
            "church": "Church",
            "enough": "Enough_Satisfied",
            "satisfied": "Enough_Satisfied",
            "friend": "Friend",
            "love": "Love",
            "me": "Me",
            "mosque": "Mosque",
            "seat": "Seat",
            "temple": "Temple",
            "you": "You"
        }
        # Ensure the data directory exists before saving
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        # Save the mapping to JSON
        with open(json_path, 'w') as f:
            json.dump(mapping, f)
        print(f"Created and saved text_to_class mapping to {json_path}")
    return mapping

# --- Initialization Function (Called by Streamlit, cached) ---
def initialize_tts_tools():
    """Loads all necessary data files for text-to-sign."""
    global labels_df, text_to_class_mapping
    labels_df = load_labels_df(LABELS_CSV)
    text_to_class_mapping = load_text_to_class_mapping(TEXT_TO_CLASS_JSON)
    print("Text-to-sign tools initialized.")
    return labels_df, text_to_class_mapping


# --- Text Processing and Video Path Finding ---
def map_text_to_classes(text, mapping, labels_df):
    """Maps input text to available sign classes."""
    text = text.lower().strip()
    words = text.split()
    found_classes = []
    for word in words:
        # Find the first matching key in the mapping for the word
        mapped_class_key = None
        for key, mapped_class in mapping.items():
             if key in word: # Simple substring match
                 mapped_class_key = mapped_class
                 break # Take the first match

        if mapped_class_key:
            # Check if the mapped class actually exists in the labels dataframe
            if mapped_class_key in labels_df['class'].unique():
                 found_classes.append(mapped_class_key)
                 print(f"Mapped '{word}' to class '{mapped_class_key}'.")
            else:
                 print(f"Warning: Mapped '{word}' to class '{mapped_class_key}', but this class not found in labels.csv.")
        else:
             print(f"No mapping found for word '{word}'.")

    return found_classes # Return a list of unique classes if preferred, but order might matter? Let's keep duplicates for now if multiple words map to same class

# --- Video Processing Functions (Adapted from Colab) ---
def read_video_frames(video_path):
    """Reads frames from a video file using OpenCV."""
    if not os.path.exists(video_path):
         print(f"Error: Video file not found at {video_path}")
         return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Ensure frame is in RGB (if needed for consistent blending, though OpenCV default is BGR)
        # For saving with cv2.VideoWriter (expects BGR), might need to keep as BGR
        # Let's keep as BGR for saving compatibility unless blending specifically requires RGB
        frames.append(frame)
    cap.release()
    return frames

def add_transition(frames1, frames2, transition_frames=10):
    """Generates frames for a fade transition between two video frame sequences."""
    if not frames1 or not frames2:
        return []

    # Use the last frame of the first video and the first frame of the second
    frame1_end = frames1[-1]
    frame2_start = frames2[0]

    # Ensure frames have the same dimensions and type
    if frame1_end.shape != frame2_start.shape or frame1_end.dtype != frame2_start.dtype:
        print(f"Warning: Frame dimensions or types mismatch for transition. Skipping transition.")
        return []

    transition = []
    for i in range(transition_frames):
        alpha = (i + 1) / (transition_frames + 1) # Alpha goes from small value to large value
        blended = cv2.addWeighted(frame1_end, (1 - alpha), frame2_start, alpha, 0)
        transition.append(blended)
    return transition

def create_combined_video(video_paths, output_dir, fps=30, transition_frames=10):
    """
    Reads multiple video files, combines their frames with transitions,
    and saves the result to a single temporary video file.
    Returns the path to the temporary file.
    """
    if not video_paths:
        return None

    all_frames = []
    video_dims = None

    # Read all video frames and transitions
    for i, video_path in enumerate(video_paths):
        frames = read_video_frames(video_path)
        if not frames:
            print(f"Skipping empty or unreadable video: {video_path}")
            continue

        # Get video dimensions from the first valid video
        if video_dims is None:
            height, width, _ = frames[0].shape
            video_dims = (width, height)

        # Append current video frames
        all_frames.extend(frames)

        # Add transition if it's not the last video and there's a next video
        if i < len(video_paths) - 1:
            next_video_path = video_paths[i+1]
            # Peak into the next video to get its first frame for transition
            cap_next = cv2.VideoCapture(next_video_path)
            ret_next, frame_next_start = cap_next.read()
            cap_next.release()

            if ret_next and video_dims and frame_next_start.shape[:2] == (video_dims[1], video_dims[0]): # Check dimensions
                 transition = add_transition(frames, [frame_next_start], transition_frames=transition_frames)
                 all_frames.extend(transition)
            else:
                 print(f"Could not create transition between {video_path} and {next_video_path}. Dimension mismatch or unreadable next video.")


    if not all_frames:
        print("No valid frames collected to create a combined video.")
        return None

    # Create a temporary directory to save the video
    temp_dir = tempfile.mkdtemp(prefix="sign_videos_")
    output_path = os.path.join(temp_dir, 'combined_sign_video.mp4')

    # Get dimensions from the first frame
    height, width, _ = all_frames[0].shape
    video_dims = (width, height)


    # Initialize VideoWriter
    # Using 'mp4v' codec for .mp4 output
    # Check if codec is available and if output can be written
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, video_dims)

    if not out.isOpened():
        print(f"Error: Could not open video writer for path {output_path} with codec mp4v.")
        print("This might be a codec issue. You might need to install gstreamer or other video libraries.")
        # Fallback or raise error? For now, print error and return None
        # Clean up temp dir before returning
        shutil.rmtree(temp_dir)
        return None


    # Write frames
    for frame in all_frames:
         # Ensure frame has the correct dimensions before writing
         if frame.shape[:2] == (height, width):
             out.write(frame)
         else:
             print(f"Warning: Skipping frame with incorrect dimensions {frame.shape[:2]} vs expected ({height}, {width})")

    out.release()

    # Return the path to the temporary file
    # Note: The temp_dir should be cleaned up later when the app closes or is refreshed
    # This is a simplification for Streamlit integration
    print(f"Combined video saved to {output_path}")
    return output_path

# --- Main function for Text-to-Sign ---
def translate_text_to_sign(text_input):
    """
    Translates text input to a sequence of sign video paths and
    creates a temporary combined video file.
    Returns the path to the temporary video file or None if mapping/video finding fails.
    """
    global labels_df, text_to_class_mapping

    if labels_df is None or text_to_class_mapping is None:
        # This should ideally be handled by Streamlit's caching calling initialize_tts_tools
        print("Error: Text-to-sign tools not initialized. Please call initialize_tts_tools first.")
        return None

    # 1. Map text to sign classes
    sign_classes = map_text_to_classes(text_input, text_to_class_mapping, labels_df)

    if not sign_classes:
        print(f"Could not map text '{text_input}' to any known sign classes.")
        return None

    # 2. Find corresponding video file paths
    video_paths = []
    # Ensure order is maintained based on the text input
    for sign_class in sign_classes:
        # Find the first video file for this class
        class_video_row = labels_df[labels_df['class'] == sign_class].iloc[:1] # Take only the first one found

        if not class_video_row.empty:
            # Construct the full path relative to the script/data location
            relative_video_path = class_video_row.iloc[0]['file_path']
            # Need to determine the base path where labels.csv lives to join correctly
            # Assuming video paths in CSV are relative to DATA_DIR
            full_video_path = os.path.join(DATA_DIR, relative_video_path) # Adjust if your CSV paths are different

            if os.path.exists(full_video_path):
                 video_paths.append(full_video_path)
                 print(f"Found video for {sign_class}: {full_video_path}")
            else:
                 print(f"Error: Video file not found at expected path: {full_video_path}")
        else:
            print(f"No video entry found in labels.csv for class: {sign_class}")


    if not video_paths:
        print(f"No valid video paths found for the mapped classes: {sign_classes}")
        return None

    # 3. Create and save the combined video
    # Using a temporary directory for the output file
    output_video_path = create_combined_video(video_paths, output_dir=tempfile.gettempdir(), fps=30, transition_frames=10)

    return output_video_path

# --- Speech Recognition Function (Requires audio file input in Streamlit) ---
def recognize_speech_from_audio(audio_file_bytes):
    """
    Performs speech recognition on audio bytes.
    Returns recognized text or None.
    """
    recognizer = sr.Recognizer()
    try:
        # Recognize speech from audio bytes (Streamlit file_uploader gives bytes)
        # Need to convert bytes to an audio format recognized by SpeechRecognition
        # A common way is to save to a temporary WAV file or use an in-memory stream
        # Saving to temp file might be simplest with current SR library
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
             tmp_audio_file.write(audio_file_bytes)
             tmp_audio_path = tmp_audio_file.name

        with sr.AudioFile(tmp_audio_path) as source:
             audio = recognizer.record(source)

        # Clean up the temporary file
        os.remove(tmp_audio_path)

        spoken_text = recognizer.recognize_google(audio)
        print(f"Recognized speech: {spoken_text}")
        return spoken_text

    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"An error occurred during speech recognition: {e}")
        return None