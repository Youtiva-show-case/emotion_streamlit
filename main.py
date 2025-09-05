import cv2
import numpy as np
import time
import requests
import io
from PIL import Image
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="🎭 Real-time Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("loaded face cascade")
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
print("loaded emotion_dict")

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

def detect_emotion_from_api(frame):
    """Send frame to emotion detection API and get results"""
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Prepare the request
        files = {
            'file': ('image.jpg', img_byte_arr, 'image/jpeg')
        }
        
        # Make API call
        response = requests.post(
            'https://emotions-from-face-production.up.railway.app/emotion/detect',
            files=files,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and 'emotion' in result:
                emotion = result['emotion']
                face_coords = result.get('face_coordinates', {})
                return emotion, face_coords
            else:
                return "Unknown", {}
        else:
            return "API Error", {}
            
    except requests.exceptions.Timeout:
        return "Timeout", {}
    except requests.exceptions.RequestException as e:
        return "Request Error", {}
    except Exception as e:
        return "Error", {}

# Function to classify emotions from detected faces using API
def classifyEmotionFromFace(face_cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Find the largest face detected
        biggest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, width, height = biggest_face
        
        try:
            # Call API for emotion detection on the whole frame
            emotion, api_face_coords = detect_emotion_from_api(frame)
            return emotion, (x, y, width, height)
        except Exception as e:
            return "Error on classification", None
    else:
        return "No face detected", None

def main():
    st.title("🎭 Real-time Emotion Detection")
    st.markdown("**Detect emotions from your camera feed using AI API!**")
    
    # Initialize session state
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Camera selection
    camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
    
    # Detection interval
    detection_interval = 2.0  # 2 seconds
    st.sidebar.info(f"API calls every {detection_interval} seconds")
    
    # Start and Stop buttons
    start_col, stop_col = st.sidebar.columns(2)
    
    with start_col:
        if st.button("Start Detection", disabled=st.session_state.detection_active):
            st.session_state.detection_active = True
    
    with stop_col:
        if st.button("Stop Detection", disabled=not st.session_state.detection_active):
            st.session_state.detection_active = False
    
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
        
        print("Camera opened. Detection started.")
        
        try:
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Check if 2 seconds has passed since last prediction (to avoid too many API calls)
                current_time = time.time()
                if current_time - last_prediction_time >= detection_interval:
                    emotion, face_coords = classifyEmotionFromFace(face_cascade, frame)
                    current_emotion = emotion
                    last_prediction_time = current_time
                    print(f"Detected emotion: {current_emotion}")
                    
                    # Add to history if it's a valid emotion
                    if emotion in emotion_colors.keys() or emotion in ["Unknown", "API Error", "Timeout", "Request Error"]:
                        st.session_state.emotion_history.append({
                            'emotion': emotion,
                            'time': time.strftime("%H:%M:%S")
                        })
                        # Keep only last 10 entries
                        if len(st.session_state.emotion_history) > 10:
                            st.session_state.emotion_history.pop(0)
                
                # Draw face detection box and emotion text only for the biggest face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Find and draw only the biggest face
                    biggest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = biggest_face
                    
                    # Draw rectangle around the biggest face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Add emotion text above the biggest face
                    cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
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
                
                # Update history display
                if st.session_state.emotion_history:
                    history_text = ""
                    for entry in reversed(st.session_state.emotion_history[-5:]):  # Show last 5
                        history_text += f"**{entry['time']}**: {entry['emotion']}\n\n"
                    history_placeholder.markdown(history_text)
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.03)
                
        except Exception as e:
            st.error(f"Error during detection: {e}")
        finally:
            cap.release()
            print("Camera released and detection stopped.")
    else:
        # Show placeholder when detection is not active
        frame_placeholder.info("Click 'Start Detection' to begin emotion detection")
        emotion_placeholder.info("No emotion detected yet")

if __name__ == "__main__":
    main()
