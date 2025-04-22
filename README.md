---
title: YOLO Vision AI
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: 3.8.0
app_file: app.py
pinned: false
---

# YOLO Vision AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0.208-yellow.svg)](https://ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-5C3EE8.svg)](https://opencv.org/)

## 📋 Description

YOLO Vision AI is a comprehensive web application built with Flask that leverages YOLO (You Only Look Once) models for various computer vision tasks. The platform provides an intuitive interface for users to upload images or videos, process them using state-of-the-art AI models, and visualize the results in real-time. The application also supports webcam streaming, allowing for live analysis of video feeds.

## ✨ Features

- **Object Detection**: Identify and locate multiple objects in images and videos with YOLOv8
- **Image Segmentation**: Generate precise pixel-level masks for detected objects
- **Pose Estimation**: Detect human body keypoints for posture and movement analysis
- **Image Classification**: Categorize images into different classes with confidence scores
- **Custom Model Training**: Train your own YOLO models with your custom datasets
- **Live Video Processing**: Analyze real-time video streams from your webcam
- **Batch Processing**: Process multiple images or video frames efficiently
- **Result Visualization**: Interactive display of detection results with bounding boxes, masks, and keypoints

## 🛠️ Technologies

- **Flask**: Backend web framework for handling requests and serving the application
- **Ultralytics YOLO**: State-of-the-art computer vision models for detection, segmentation, and more
- **PyTorch**: Deep learning framework powering the AI models
- **OpenCV**: Computer vision library for image and video processing
- **Bootstrap 5**: Frontend framework for responsive and modern UI
- **Docker**: Containerization for easy deployment and scalability

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-vision-ai.git
cd yolo-vision-ai

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Using Docker:
```bash
# Build the Docker image
docker build -t yolo-vision-ai .

# Run the container
docker run -p 5000:5000 yolo-vision-ai
```

## 🚀 Usage

### Web Interface

1. **Access the Application**: Open your browser and navigate to `http://localhost:5000`
2. **Select AI Task**: Choose from detection, segmentation, pose estimation, or classification
3. **Upload Media**: Upload an image/video or provide a URL
4. **Adjust Parameters**: Set the confidence threshold as needed
5. **Process and View**: Visualize the results with interactive overlays

### Custom Model Training

1. Navigate to the Training page
2. Upload your dataset in ZIP format (must follow YOLO dataset structure)
3. Configure training parameters (epochs, batch size, image size)
4. Start training and monitor progress
5. Download your trained model for future use

### Video Streaming

1. Go to the Video Stream page
2. Allow camera access when prompted
3. Select the processing type and confidence threshold
4. Start the stream to see real-time AI analysis
5. Capture frames as needed for further analysis

## 📸 Computer Vision Capabilities

### Object Detection
- Detect multiple object classes simultaneously
- Adjustable confidence thresholds
- Bounding box visualization with class labels

### Image Segmentation
- Pixel-perfect object masks
- Instance segmentation for multiple objects
- Colored mask visualization

### Pose Estimation
- Human body keypoint detection
- Joint connections visualization
- Support for multiple people in the same frame

### Classification
- Multi-class image categorization
- Confidence scores for each category
- Support for custom classification models

## 🔧 Project Structure

- `app.py`: Main application file with routes and processing logic
- `utils/`: Helper modules for YAML configuration, model utilities, and metrics
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and static assets
- `static/models/`: Pre-trained and custom-trained model storage
- `static/uploads/`: Temporary storage for uploaded media
- `static/results/`: Processed images and results
- `static/datasets/`: Training datasets

## 🌐 API Endpoints

The application provides several REST API endpoints for programmatic access:

- `/api/process-image`: Process an uploaded image
- `/api/process-video`: Process an uploaded video
- `/api/start-training`: Start training a custom model
- `/api/training-status`: Check training progress
- `/api/start-stream`: Start video stream processing
- `/api/stop-stream`: Stop video stream
- `/api/list-models`: List available models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Ultralytics team for the YOLO models and implementation
- Flask team for the lightweight web framework
- PyTorch community for the deep learning tools
- OpenCV contributors for computer vision algorithms