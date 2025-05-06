# app.py
import streamlit as st
from PIL import Image
import io
import os
import shutil
import tempfile # Import tempfile for path handling

# --- Set Streamlit Page Configuration ---
# This MUST be the FIRST Streamlit command called.
# Using 'wide' layout for the main columns, we'll manage content width using containers.
st.set_page_config(
    page_title="Sign Translate",
    layout="wide",
    initial_sidebar_state="collapsed",
    
    # icon="👋" # Keep this commented out if your Streamlit version doesn't support it
)

# --- Import functions and variables from your tools ---
# Ensure inference_tool.py and text_to_sign_tool.py are in the same directory
try:
    from inference_tool import (
        load_inference_model, load_class_names, predict_sign,
        IMG_HEIGHT, IMG_WIDTH, MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH, METADATA_PATH
    )
    # print("Sign-to-text tools imported successfully.")
except ImportError as e:
    st.error(f"Error importing Sign-to-Text tool: {e}. Please ensure 'inference_tool.py' and model files are in the correct directory.")
    st.stop() # Stop the app if core tools can't be imported
except Exception as e:
    st.error(f"An unexpected error occurred loading Sign-to-Text tool: {e}")
    st.stop()

try:
    from text_to_sign_tool import (
        initialize_tts_tools, translate_text_to_sign, recognize_speech_from_audio,
        DATA_DIR # We need DATA_DIR to show the user where data is expected
    )
    # print("Text-to-Sign tools imported successfully.")
except ImportError as e:
    st.error(f"Error importing Text-to-Sign tool: {e}. Please ensure 'text_to_sign_tool.py' is in the same directory.")
    st.stop() # Stop the app if core tools can't be imported
except Exception as e:
    st.error(f"An unexpected error occurred loading Text-to-Sign tool: {e}")
    st.stop()


# --- Cached Resource Loading ---
# Use st.cache_resource to load resources only once across all sessions
# (Suitable for large models/datasets that don't change during runtime)

@st.cache_resource
def cached_load_sign_to_text_model(config_path, weights_path):
    """Caches the loading of the Keras model."""
    print("Attempting to load Sign-to-Text model...")
    try:
        model = load_inference_model(config_path, weights_path)
        print("Sign-to-Text model loaded and cached.")
        return model
    except Exception as e:
        st.error(f"Failed to load the Sign-to-Text model: {e}. Check {MODEL_CONFIG_PATH} and {MODEL_WEIGHTS_PATH}.")
        st.stop() # Critical failure

@st.cache_resource
def cached_load_sign_to_text_class_names(metadata_path):
    """Caches the loading of class names for sign-to-text."""
    print("Attempting to load Sign-to-Text class names...")
    try:
        class_names = load_class_names(metadata_path)
        print("Sign-to-Text class names loaded and cached.")
        return class_names
    except Exception as e:
        st.error(f"Failed to load Sign-to-Text class names from {METADATA_PATH}: {e}.")
        st.stop() # Critical failure


@st.cache_resource
def cached_initialize_text_to_sign_tools():
    """Caches the initialization of text-to-sign data (labels, mapping)."""
    print("Attempting to initialize Text-to-Sign tools...")
    try:
        # This calls the function in text_to_sign_tool.py
        labels_df, text_to_class_mapping = initialize_tts_tools()
        print("Text-to-Sign tools initialized and cached.")
        return labels_df, text_to_class_mapping
    except Exception as e:
        st.error(f"Failed to initialize Text-to-Sign tools. Please check data files in '{DATA_DIR}': {e}")
        st.stop() # Critical failure


# --- Load all necessary resources when the app starts ---
# These functions are called once and their results cached
sign_to_text_model = cached_load_sign_to_text_model(MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH)
sign_to_text_class_names = cached_load_sign_to_text_class_names(METADATA_PATH)
# labels_df and text_to_class_mapping are loaded and cached internally by text_to_sign_tool
# via the initialize_tts_tools function called here:
cached_initialize_text_to_sign_tools()


# --- Session State Management ---
# Initialize session state variables
if 'mode' not in st.session_state:
    st.session_state.mode = 'sign_to_text' # Default mode
if 'tts_text_input' not in st.session_state:
     st.session_state.tts_text_input = "" # Store text input for Text-to-Sign
if 'tts_generated_video_path' not in st.session_state:
     st.session_state.tts_generated_video_path = None # Store path of the generated video

# --- Swap Button Logic ---
def toggle_mode():
    """Toggles between 'sign_to_text' and 'text_to_sign' modes."""
    # Clean up previous text input and generated video path when swapping
    st.session_state.tts_text_input = ""
    st.session_state.tts_generated_video_path = None

    if st.session_state.mode == 'sign_to_text':
        st.session_state.mode = 'text_to_sign'
        st.info("Switched to **Text/Speech to Sign Language** mode.")
    else:
        st.session_state.mode = 'sign_to_text'
        st.info("Switched to **Sign Language to Text/Speech** mode.")

# --- App Title and Swap Button ---

st.title("Sign Translate")
st.write("Translate between Sign Language and Text/Speech.")

# Place the swap button below the title, potentially centered
button_col1, button_col2, button_col3 = st.columns([1, 2, 1]) # Adjust column ratios for centering
with button_col2:
    swap_button = st.button(
        "🔄 Swap Translation Direction",
        on_click=toggle_mode, # Use on_click to toggle mode
        use_container_width=True,
        key="swap_button"
    )

st.markdown("---") # Visual separator


# --- Main Content Area (Dynamic based on Mode) ---

# Use containers within columns to give content a framed look
col1, col2 = st.columns(2)

# --- UI for Sign Language to Text/Speech Mode ---
if st.session_state.mode == 'sign_to_text':
    st.subheader("Current Mode: Sign Language to Text/Speech")

    # --- Column 1: Input (Sign Image/Webcam) ---
    with col1:
        st.markdown("#### Input (Sign Image)") # Use markdown for heading style

        with st.container(border=True): # Bordered container for input controls
            st.write("###### Choose your input source:")
            input_method = st.radio(
                "", ('Upload Image', 'Capture from Webcam'),
                key="stt_input_method", horizontal=True, help="Select how you want to provide the sign image."
            )

        image_input = None # Variable to hold the image data

        with st.container(border=True): # Bordered container for image display (acts as frame)
             st.markdown("###### Image Preview:")
             if input_method == 'Upload Image':
                 uploaded_file = st.file_uploader(
                     "Upload an image file", # Clearer label
                     type=["jpg", "png", "jpeg"],
                     key="stt_uploader",
                     help="Upload a static image of a sign."
                 )
                 if uploaded_file is not None:
                     image_input = Image.open(uploaded_file)
                     st.image(image_input, caption="Uploaded Image", use_column_width=True)
                 else:
                     st.info("Please upload a .jpg, .png, or .jpeg image file in this section.")


             elif input_method == 'Capture from Webcam':
                 camera_image = st.camera_input(
                     "Take a picture using your camera", # Clearer label
                     key="stt_camera",
                     help="Capture a single frame from your webcam."
                 )
                 if camera_image is not None:
                     image_input = Image.open(camera_image) # Streamlit camera_input directly gives BytesIO
                     st.image(image_input, caption="Captured Image", use_column_width=True)
                 else:
                     st.info("Use the button above to capture an image from your webcam.")

             # You could add a "Process Image" button here if you don't want it to process immediately on upload/capture
             # if image_input and st.button("Analyze Sign"):
             #     # Trigger prediction logic below
             #     pass # Logic is already triggered by `if image_input is not None` check below

    # --- Column 2: Output (Text/Speech) ---
    with col2:
        st.markdown("#### Output (Text/Speech)") # Use markdown for heading style

        with st.container(border=True): # Bordered container for output type selection
            st.write("###### Choose your output format:")
            stt_output_type = st.radio(
                "", ('Text Output', 'Speech Output (Placeholder)'),
                key="stt_output_type", horizontal=True, help="Select how you want the translation displayed."
            )

        with st.container(border=True): # Bordered container for prediction results
            st.markdown("###### Translation Results:")
            if image_input is None:
                 st.write("Prediction results will appear here after you provide an image in the left column.")
            else:
                # Add a spinner while processing
                with st.spinner("Analyzing sign..."):
                    try:
                        predicted_class, confidence, all_predictions = predict_sign(sign_to_text_model, image_input, sign_to_text_class_names)

                        st.subheader("Prediction:") # Main result heading

                        if stt_output_type == 'Text Output':
                            # Use st.metric for key result
                            st.metric(label="Predicted Sign", value=predicted_class, delta=f"{confidence:.1%} Confidence")

                            # Optional: Display all probabilities in an expander
                            with st.expander("View detailed probabilities"):
                                 st.write("Probabilities for each class:")
                                 sorted_preds = sorted(zip(sign_to_text_class_names, all_predictions), key=lambda item: item[1], reverse=True)
                                 for class_name, prob in sorted_preds:
                                     st.write(f"- **{class_name}**: {prob:.4f}") # Bold class names

                        elif stt_output_type == 'Speech Output (Placeholder)':
                            st.write(f"**Predicted Sign (Text):** {predicted_class}") # Still show the text prediction
                            st.write(f"**Confidence:** {confidence:.2f}")
                            st.info(f"Speech output for '{predicted_class}' would play here. (Feature not yet implemented)")
                            # Add placeholder button if needed
                            # if st.button(f"🔊 Speak '{predicted_class}' (Placeholder)", key="speak_stt"):
                            #     st.write("*(Speech playback not yet implemented)*")


                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.write("Please try another image or input method.")


# --- UI for Text/Speech to Sign Language Mode ---
elif st.session_state.mode == 'text_to_sign':
    st.subheader("Current Mode: Text/Speech to Sign Language")

    # --- Column 1: Input (Text/Speech) ---
    with col1:
        st.markdown("#### Input (Text/Speech)")

        with st.container(border=True): # Bordered container for input type selection
             st.write("###### Choose your input format:")
             tts_input_type = st.radio(
                 "", ('Type Text', 'Upload Audio (Placeholder)'), # Added audio upload placeholder
                 key="tts_input_type", horizontal=True, help="Select how you want to provide text or speech input."
             )

             # Use session state to manage the text input value
             text_input_value = st.session_state.tts_text_input

             if tts_input_type == 'Type Text':
                 # Update session state directly when the text input changes
                 text_input_value = st.text_input(
                     "Enter phrase here (e.g., 'I love you')",
                     value=st.session_state.tts_text_input, # Set initial value from state
                     key="text_to_sign_input",
                     help="Type the phrase you want to translate into sign language."
                 )
                 # Update state if text input widget changed
                 st.session_state.tts_text_input = text_input_value

                 # Button to trigger translation
                 process_text_button = st.button(
                     "Translate Text to Sign",
                     key="process_text_button",
                     help="Click to generate the sign video for the entered text."
                 )

                 # Trigger video generation only when the button is clicked AND there's text
                 if process_text_button and st.session_state.tts_text_input:
                     # Set the state variable that signals video generation is requested
                     # We'll clear the previous video path if any
                     st.session_state.tts_generated_video_path = None # Clear old path before generating new
                     # Use the text from session state
                     text_to_process = st.session_state.tts_text_input
                     with st.spinner(f"Generating sign video for '{text_to_process}'..."):
                        try:
                            # Call the function from text_to_sign_tool.py
                            video_output_path = translate_text_to_sign(text_to_process)

                            if video_output_path:
                                 st.session_state.tts_generated_video_path = video_output_path # Store the new path in state
                                 st.success("Video generation initiated. See output on the right.")
                            else:
                                st.warning(f"Could not generate video for phrase: '{text_to_process}'. No corresponding signs found or video creation failed.")
                                st.session_state.tts_generated_video_path = None # Ensure state is clear on failure

                        except Exception as e:
                            st.error(f"An error occurred during video generation: {e}")
                            st.write("Please try a different phrase.")
                            st.session_state.tts_generated_video_path = None # Ensure state is clear on failure


             elif tts_input_type == 'Upload Audio (Placeholder)':
                  st.info("Audio upload and speech recognition for Text-to-Sign is a placeholder feature.")
                  st.warning("Currently, only 'Type Text' input is functional in this mode.")
                  # Add placeholder file uploader
                  # audio_file = st.file_uploader(
                  #      "Upload an audio file (.wav)", # Specify format expected by SpeechRecognition
                  #      type=["wav"],
                  #      key="tts_audio_uploader",
                  #      help="Upload a .wav file containing spoken words (feature not yet implemented)."
                  # )
                  # # This part needs implementation for speech recognition and triggering translation
                  # if audio_file is not None:
                  #     st.write("Processing audio file...")
                  #     # Add speech recognition logic here...
                  #     # recognized_text = recognize_speech_from_audio(audio_file.getvalue())
                  #     # if recognized_text:
                  #     #    st.write(f"Recognized Text: **{recognized_text}**")
                  #     #    # Then trigger translation based on recognized_text
                  #     # else:
                  #     #    st.warning("Could not recognize speech from audio.")

    # --- Column 2: Output (Sign Video) ---
    with col2:
        st.markdown("#### Output (Sign Video)")

        with st.container(border=True): # Bordered container for video display
            st.markdown("###### Generated Video:")
            # Display the video if a path is stored in session state and the file exists
            if st.session_state.tts_generated_video_path and os.path.exists(st.session_state.tts_generated_video_path):
                 try:
                     st.video(st.session_state.tts_generated_video_path) # Display the temporary video file
                     st.success("Video displayed below.")

                     # Add a download button for the video
                     with open(st.session_state.tts_generated_video_path, "rb") as file:
                         st.download_button(
                             label="Download Video",
                             data=file,
                             file_name="translated_sign_video.mp4",
                             mime="video/mp4",
                             key="download_tts_video",
                             help="Download the generated sign language video."
                         )
                 except Exception as e:
                     st.error(f"Error displaying video: {e}. The video file might be corrupted or the player encountered an issue.")
                     # Clear the state so it doesn't try to display the bad file again
                     st.session_state.tts_generated_video_path = None

            else:
                 # Message when no video has been generated yet
                 if st.session_state.tts_text_input and st.session_state.mode == 'text_to_sign':
                      # If text is entered but no video generated yet (e.g., waiting for button click)
                      st.write("Enter text and click 'Translate Text to Sign' to generate a video.")
                 else:
                      st.write("Enter text or provide audio in the left column to generate a sign language video.")


# --- Footer ---
st.markdown("---")
st.markdown("App built with Streamlit and TensorFlow")
st.markdown(f"Sign video data expected in the '{DATA_DIR}' directory relative to the app scripts.")
st.markdown("Model: Transfer Learning with ResNet50")

# --- Temporary Directory Cleanup Note ---
# In a real deployment, you would need a robust way to clean up the /tmp directories created by tempfile.mkdtemp
# Streamlit doesn't provide built-in temp file management across sessions.
# For local testing, the OS usually cleans up /tmp on restart.
st.info(f"Generated videos for Text-to-Sign are saved to temporary directories (e.g., in {tempfile.gettempdir()}). These directories are not automatically cleaned up by the app itself.")


# --- Potential Cleanup on App Shutdown (Advanced) ---
# Streamlit does not have a reliable shutdown hook.
# Cleaning up temporary files is typically handled by:
# 1. The OS clearing the /tmp directory on restart.
# 2. A separate cleanup process or script if deploying long-term.
# 3. Designing the app to use a non-temp directory that can be manually cleared.