import streamlit as st
import cv2
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

# Configure Streamlit page
st.set_page_config(
    page_title="🎭 Real-time Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

@st.cache_resource
def load_model_and_cascade():
    """Load the trained model and face cascade classifier"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        trained_model = load_model("trained_models/model_tf_e.h5")
        return trained_model, face_cascade
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Mapping of emotion indices to emotion labels
emotion_dict = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happy',
    4: 'Sad',
    5: 'Angry',
    6: 'Neutral'
}

# Define emotion colors for visual feedback
emotion_colors = {
    'Surprise': '#FFD700',
    'Fear': '#8A2BE2',
    'Disgust': '#32CD32',
    'Happy': '#FF69B4',
    'Sad': '#4169E1',
    'Angry': '#FF4500',
    'Neutral': '#808080'
}

def classifyEmotionFromFace(trained_model, face_cascade, frame):
    """Classify emotions from detected faces"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Find the largest face detected
        biggest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, width, height = biggest_face
        
        try:
            # Extract and preprocess the face
            face = frame[y:y+height, x:x+width]
            face = cv2.resize(face, (100, 100))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_array = np.expand_dims(face, axis=0)
            face_array = preprocess_input(face_array)
            face_array = face_array / 255.0

            # Predict the emotion
            prediction = trained_model.predict(face_array, verbose=0)
            maxindex = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            return emotion_dict[maxindex], (x, y, width, height), confidence
        except Exception as e:
            return "Error", None, 0.0
    else:
        return "No face detected", None, 0.0

def main():
    st.title("🎭 Real-time Emotion Detection")
    st.markdown("**Detect emotions from your camera feed in real-time!**")
    
    # Load model and cascade
    trained_model, face_cascade = load_model_and_cascade()
    
    if trained_model is None or face_cascade is None:
        st.error("Failed to load model or face cascade. Please check if the model file exists at 'trained_models/model_tf_e.h5'")
        return
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Camera selection
    camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
    
    # Detection interval
    detection_interval = st.sidebar.slider(
        "Detection Interval (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="How often to run emotion detection (lower = more frequent but slower)"
    )
    
    # Start/Stop detection
    if st.sidebar.button("Start Detection" if not st.session_state.detection_active else "Stop Detection"):
        st.session_state.detection_active = not st.session_state.detection_active
    
    # Clear history
    if st.sidebar.button("Clear History"):
        st.session_state.emotion_history = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        frame_placeholder = st.empty()
        
    with col2:
        st.subheader("Current Emotion")
        emotion_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        st.subheader("Emotion History")
        history_placeholder = st.empty()
    
    # Camera processing
    if st.session_state.detection_active:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Cannot open camera {camera_index}")
            return
        
        last_prediction_time = time.time()
        current_emotion = "Starting..."
        current_confidence = 0.0
        
        try:
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Check if it's time for a new prediction
                current_time = time.time()
                if current_time - last_prediction_time >= detection_interval:
                    emotion, face_coords, confidence = classifyEmotionFromFace(trained_model, face_cascade, frame)
                    current_emotion = emotion
                    current_confidence = confidence
                    last_prediction_time = current_time
                    
                    # Add to history if it's a valid emotion
                    if emotion in emotion_dict.values():
                        st.session_state.emotion_history.append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'time': time.strftime("%H:%M:%S")
                        })
                        # Keep only last 10 entries
                        if len(st.session_state.emotion_history) > 10:
                            st.session_state.emotion_history.pop(0)
                
                # Draw face detection and emotion on frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Find and draw only the biggest face
                    biggest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = biggest_face
                    
                    # Draw rectangle around the biggest face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add emotion text above the biggest face
                    cv2.putText(frame, f"{current_emotion}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {current_confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Update emotion display
                if current_emotion in emotion_colors:
                    emotion_placeholder.markdown(
                        f"<div style='background-color: {emotion_colors[current_emotion]}; "
                        f"padding: 20px; border-radius: 10px; text-align: center; color: white; font-size: 24px; font-weight: bold;'>"
                        f"{current_emotion}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    emotion_placeholder.info(current_emotion)
                
                confidence_placeholder.metric("Confidence", f"{current_confidence:.2%}")
                
                # Update history display
                if st.session_state.emotion_history:
                    history_text = ""
                    for entry in reversed(st.session_state.emotion_history[-5:]):  # Show last 5
                        history_text += f"**{entry['time']}**: {entry['emotion']} ({entry['confidence']:.1%})\n\n"
                    history_placeholder.markdown(history_text)
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.03)
                
        except Exception as e:
            st.error(f"Error during detection: {e}")
        finally:
            cap.release()
    else:
        # Show placeholder when detection is not active
        frame_placeholder.info("Click 'Start Detection' to begin emotion detection")
        emotion_placeholder.info("No emotion detected yet")

if __name__ == "__main__":
    main()
