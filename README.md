# 🎭 Real-time Emotion Detection with Streamlit

A Streamlit web application that performs real-time emotion detection using your camera feed. The app uses an AI API service to classify emotions from facial expressions in real-time.

## Features

- **Real-time emotion detection** from camera feed via AI API
- **7 emotion categories**: Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral
- **Visual feedback** with colored emotion indicators
- **Face detection** with bounding boxes
- **Emotion history** tracking
- **API-based processing** for accurate results
- **Multiple camera support**

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run main.py
```

## Usage

1. **Start the app**: Run the command above and open the provided URL in your browser
2. **Camera setup**: Select your camera from the dropdown in the sidebar
3. **Start detection**: Click the "Start Detection" button in the sidebar
4. **Position yourself**: Make sure your face is visible in the camera feed
5. **View results**: Watch as the app detects and classifies your emotions via AI API calls

## Technical Details

### API Integration

The app uses an external AI API for emotion detection:
- API endpoint: `https://emotions-from-face-production.up.railway.app/emotion/detect`
- Sends camera frames as JPEG images to the API
- Receives emotion predictions with 7 categories: Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral
- API calls are made every 2 seconds to optimize performance

### Face Detection

- Uses OpenCV's Haar Cascade classifier for face detection
- Focuses on the largest detected face when multiple faces are present
- Draws bounding boxes around detected faces

### Performance

- API calls every 2 seconds to balance accuracy and performance
- Runs at approximately 30 FPS for smooth video display
- Handles API timeouts and errors gracefully

## Troubleshooting

### Camera Issues
- Try different camera indices (0, 1, 2) if your camera isn't detected
- Ensure camera permissions are granted to your browser
- Check that no other applications are using the camera

### API Issues
- Check your internet connection
- Verify the API endpoint is accessible
- Monitor for API timeout errors in the app

### Performance Issues
- The app automatically limits API calls to every 2 seconds
- Close other resource-intensive applications
- Ensure stable internet connection for API calls

## Dependencies

- `streamlit`: Web app framework
- `opencv-python`: Computer vision and camera capture
- `requests`: HTTP client for API calls
- `numpy`: Numerical computations
- `Pillow`: Image processing support

## License

This project is open source and available under the MIT License. 