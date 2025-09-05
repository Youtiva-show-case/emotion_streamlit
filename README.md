# 🎭 Real-time Emotion Detection with Streamlit

A Streamlit web application that performs real-time emotion detection using your camera feed. The app uses computer vision and deep learning to classify emotions from facial expressions.

## Features

- **Real-time emotion detection** from camera feed
- **7 emotion categories**: Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral
- **Visual feedback** with colored emotion indicators
- **Face detection** with bounding boxes
- **Emotion history** tracking
- **Adjustable detection intervals** for performance optimization
- **Multiple camera support**

## Setup Instructions

### 1. Install Dependencies

**Option A: Use Compatible Version (Recommended)**
```bash
pip install -r requirements_compatible.txt
```

**Option B: Use Latest Version (May require compatibility fixes)**
```bash
pip install -r requirements.txt
```

> **Note:** Your model was created with Keras 2.10.0. For best compatibility, use Option A.

### 2. Add Your Trained Model

Place your trained emotion classification model in the `trained_models/` directory:
- The model file should be named `model_tf_e.h5`
- Path: `trained_models/model_tf_e.h5`

### 3. Run the Application

```bash
streamlit run main.py
```

## Usage

1. **Start the app**: Run the command above and open the provided URL in your browser
2. **Camera setup**: Select your camera from the dropdown in the sidebar
3. **Adjust settings**: Use the detection interval slider to balance performance and accuracy
4. **Start detection**: Click the "Start Detection" button in the sidebar
5. **Position yourself**: Make sure your face is visible in the camera feed
6. **View results**: Watch as the app detects and classifies your emotions in real-time

## Technical Details

### Model Requirements

The app expects a TensorFlow/Keras model (`model_tf_e.h5`) that:
- Takes 100x100 RGB images as input
- Outputs predictions for 7 emotion classes
- Uses the emotion mapping: {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}

### Face Detection

- Uses OpenCV's Haar Cascade classifier for face detection
- Focuses on the largest detected face when multiple faces are present
- Draws bounding boxes around detected faces

### Performance

- Configurable detection intervals (0.1 to 2.0 seconds)
- Runs at approximately 30 FPS for smooth video display
- Uses Streamlit's caching for efficient model loading

## Troubleshooting

### Camera Issues
- Try different camera indices (0, 1, 2) if your camera isn't detected
- Ensure camera permissions are granted to your browser
- Check that no other applications are using the camera

### Model Issues
- Verify the model file exists at `trained_models/model_tf_e.h5`
- Ensure the model is compatible with TensorFlow/Keras
- Check that the model expects 100x100 RGB input images

### Performance Issues
- Increase the detection interval for better performance
- Close other resource-intensive applications
- Use a lower resolution camera if available

## Dependencies

- `streamlit`: Web app framework
- `opencv-python`: Computer vision and camera capture
- `tensorflow`: Deep learning model inference
- `numpy`: Numerical computations
- `Pillow`: Image processing support

## License

This project is open source and available under the MIT License. 