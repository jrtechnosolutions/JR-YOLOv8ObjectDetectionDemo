import os
import io
import yaml
import uuid
import zipfile
import shutil
import threading
import time
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_file, flash, session
from werkzeug.utils import secure_filename
import requests
from ultralytics import YOLO

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODELS_FOLDER'] = 'models'
app.config['DATASETS_FOLDER'] = 'datasets'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'zip'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], 
               app.config['RESULTS_FOLDER'], 
               app.config['MODELS_FOLDER'], 
               app.config['DATASETS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global variables
camera = None
camera_active = False
global_frame = None
training_thread = None
is_training = False
training_status = {"progress": 0, "message": "", "complete": False}

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename(filename):
    """Generate a unique filename preserving the original extension"""
    ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
    return f"{uuid.uuid4().hex}.{ext}" if ext else f"{uuid.uuid4().hex}"

def load_model(model_type):
    """Load appropriate YOLO model based on the requested type"""
    models = {
        'detection': 'yolov8n.pt',
        'segmentation': 'yolov8n-seg.pt',
        'pose': 'yolov8n-pose.pt',
        'classification': 'yolov8n-cls.pt'
    }
    
    model_path = os.path.join(app.config['MODELS_FOLDER'], models[model_type])
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        model = YOLO(models[model_type])
        model.save(model_path)
    else:
        model = YOLO(model_path)
    
    return model

def process_image(image_path, model_type, confidence=0.25):
    """Process an image with selected YOLO model"""
    model = load_model(model_type)
    results = model(image_path, conf=confidence)
    
    # Save the result image
    if model_type == 'classification':
        # For classification, we'll return class probabilities
        probs = results[0].probs.data.tolist()
        names = results[0].names
        result_data = {names[i]: probs[i] for i in range(len(names))}
        return result_data, None
    else:
        # For other models, we'll save and return the plotted image
        result_img = results[0].plot()
        result_filename = f"{generate_unique_filename('result')}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Convert from BGR to RGB for saving with PIL
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(result_img_rgb).save(result_path)
        
        # Extract result data
        if model_type == 'detection':
            boxes = results[0].boxes
            result_data = []
            for i, box in enumerate(boxes):
                cls = int(box.cls[0].item())
                name = results[0].names[cls]
                conf = round(box.conf[0].item(), 2)
                xyxy = box.xyxy[0].tolist()
                result_data.append({
                    "class": name,
                    "confidence": conf,
                    "box": xyxy
                })
        elif model_type == 'segmentation':
            masks = results[0].masks
            result_data = []
            if masks is not None:
                for i, mask in enumerate(masks):
                    cls = int(results[0].boxes[i].cls[0].item())
                    name = results[0].names[cls]
                    conf = round(results[0].boxes[i].conf[0].item(), 2)
                    result_data.append({
                        "class": name,
                        "confidence": conf
                    })
        elif model_type == 'pose':
            keypoints = results[0].keypoints
            result_data = []
            if keypoints is not None:
                for i in range(len(keypoints)):
                    result_data.append({
                        "keypoints": keypoints[i].data[0].tolist()
                    })
        
        return result_data, result_filename

def process_from_url(image_url, model_type, confidence=0.25):
    """Process an image from URL"""
    try:
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))
        
        # Save the image temporarily
        temp_filename = generate_unique_filename('temp')
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        img.save(temp_path)
        
        # Process the image
        result_data, result_filename = process_image(temp_path, model_type, confidence)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return result_data, result_filename
    except Exception as e:
        return {"error": str(e)}, None

def start_training(dataset_path, epochs, model_type, batch_size, img_size):
    """Start training process in a separate thread"""
    global is_training, training_status
    is_training = True
    training_status = {"progress": 0, "message": "Preparing training...", "complete": False}
    
    # Create a new thread for training
    training_thread = threading.Thread(
        target=run_training,
        args=(dataset_path, epochs, model_type, batch_size, img_size)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return training_thread

def run_training(dataset_path, epochs, model_type, batch_size, img_size):
    """Run YOLO training on the provided dataset"""
    global training_status
    
    try:
        # Create YAML configuration file for training
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        yaml_path = os.path.join(app.config['DATASETS_FOLDER'], f"{dataset_name}.yaml")
        
        # Basic YAML configuration
        yaml_config = {
            'path': os.path.abspath(os.path.join(app.config['DATASETS_FOLDER'], dataset_name)),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {}  # Will be populated based on classes.txt if available
        }
        
        # Check if classes.txt exists
        classes_path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
                for i, cls in enumerate(classes):
                    yaml_config['names'][i] = cls
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f)
        
        # Select model based on model_type
        model_map = {
            'detection': 'yolov8n.pt',
            'segmentation': 'yolov8n-seg.pt',
            'pose': 'yolov8n-pose.pt',
            'classification': 'yolov8n-cls.pt'
        }
        
        model = YOLO(model_map[model_type])
        
        # Start training
        training_status["message"] = f"Training started. Model: {model_type}, Epochs: {epochs}"
        
        # Custom callback to update progress
        def on_train_epoch_end(trainer):
            global training_status
            epoch = trainer.epoch
            training_status["progress"] = int((epoch / epochs) * 100)
            training_status["message"] = f"Training: {epoch}/{epochs} epochs completed"
        
        # Register callback
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        # Start training
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=app.config['MODELS_FOLDER'],
            name=f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Save the trained model
        model_save_path = os.path.join(
            app.config['MODELS_FOLDER'], 
            f"trained_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        model.save(model_save_path)
        
        # Convert to ONNX
        onnx_path = model_save_path.replace('.pt', '.onnx')
        model.export(format="onnx", save=True)
        
        # Update status
        training_status["progress"] = 100
        training_status["message"] = f"Training complete. Model saved at {model_save_path}"
        training_status["complete"] = True
        training_status["model_path"] = model_save_path
        training_status["onnx_path"] = onnx_path
        
    except Exception as e:
        training_status["message"] = f"Training error: {str(e)}"
        training_status["complete"] = True
    
    finally:
        global is_training
        is_training = False

def extract_dataset(zip_path):
    """Extract dataset from zip file and organize it properly"""
    dataset_name = os.path.basename(zip_path).split('.')[0]
    extract_path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name)
    
    # Create extraction directory
    os.makedirs(extract_path, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Check dataset structure and organize if needed
    # This handles common formats like YOLO, COCO, etc.
    required_dirs = ['train', 'valid', 'test']
    
    # If the dataset is already properly organized
    if any(os.path.exists(os.path.join(extract_path, d)) for d in required_dirs):
        return extract_path
    
    # Simple auto-organization: look for annotations and images
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    annotation_extensions = {'.txt', '.xml', '.json'}
    
    image_files = []
    annotation_files = []
    
    for root, _, files in os.walk(extract_path):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext in img_extensions:
                image_files.append(path)
            elif ext in annotation_extensions:
                annotation_files.append(path)
    
    # If we found images but no organized structure, create one
    if image_files and not any(os.path.exists(os.path.join(extract_path, d)) for d in required_dirs):
        # Create basic train/valid split (80/20)
        os.makedirs(os.path.join(extract_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'valid', 'labels'), exist_ok=True)
        
        # Shuffle and split files
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.8)
        train_images = image_files[:split_idx]
        valid_images = image_files[split_idx:]
        
        # Copy files to appropriate locations
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(extract_path, 'train', 'images', os.path.basename(img_path)))
            
            # Find corresponding annotation if exists
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for ann_path in annotation_files:
                if os.path.splitext(os.path.basename(ann_path))[0] == base_name:
                    shutil.copy(ann_path, os.path.join(extract_path, 'train', 'labels', 
                                                    os.path.basename(ann_path)))
                    break
        
        for img_path in valid_images:
            shutil.copy(img_path, os.path.join(extract_path, 'valid', 'images', os.path.basename(img_path)))
            
            # Find corresponding annotation if exists
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for ann_path in annotation_files:
                if os.path.splitext(os.path.basename(ann_path))[0] == base_name:
                    shutil.copy(ann_path, os.path.join(extract_path, 'valid', 'labels', 
                                                    os.path.basename(ann_path)))
                    break
    
    return extract_path

# Video streaming functions
def get_video_stream(model_type, confidence=0.25):
    """Generator function for video streaming"""
    global camera, camera_active, global_frame
    
    # Load model
    model = load_model(model_type)
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame with YOLO
        results = model(frame, conf=confidence)
        processed_frame = results[0].plot()
        
        # Store the processed frame
        global_frame = processed_frame
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the output
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release the camera when done
    if camera is not None:
        camera.release()

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/detection')
def detection():
    """Object detection page"""
    return render_template('detection.html')

@app.route('/segmentation')
def segmentation():
    """Image segmentation page"""
    return render_template('segmentation.html')

@app.route('/pose')
def pose():
    """Pose estimation page"""
    return render_template('pose.html')

@app.route('/classification')
def classification():
    """Image classification page"""
    return render_template('classification.html')

@app.route('/training')
def training():
    """Model training page"""
    return render_template('training.html')

@app.route('/streaming')
def streaming():
    """Video streaming page"""
    return render_template('streaming.html')

@app.route('/api/process-image', methods=['POST'])
def api_process_image():
    """API endpoint for processing an image"""
    # Check if image file is provided
    if 'image' not in request.files and 'url' not in request.form:
        return jsonify({'error': 'No image file or URL provided'}), 400
    
    # Get parameters
    model_type = request.form.get('model_type', 'detection')
    confidence = float(request.form.get('confidence', 0.25))
    
    result_data = None
    result_filename = None
    
    # Process from URL if provided
    if 'url' in request.form and request.form['url']:
        image_url = request.form['url']
        result_data, result_filename = process_from_url(image_url, model_type, confidence)
    
    # Process uploaded file
    elif 'image' in request.files:
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            result_data, result_filename = process_image(filepath, model_type, confidence)
    
    # Return the results
    if result_filename:
        return jsonify({
            'status': 'success',
            'result_data': result_data,
            'result_image': url_for('results', filename=result_filename)
        })
    elif result_data:
        return jsonify({
            'status': 'success',
            'result_data': result_data
        })
    else:
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/api/start-training', methods=['POST'])
def api_start_training():
    """API endpoint to start model training"""
    global is_training, training_thread
    
    # Check if already training
    if is_training:
        return jsonify({
            'status': 'error',
            'message': 'Training already in progress'
        }), 400
    
    # Check if dataset is provided
    if 'dataset' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No dataset file provided'
        }), 400
    
    # Get parameters
    epochs = int(request.form.get('epochs', 10))
    model_type = request.form.get('model_type', 'detection')
    batch_size = int(request.form.get('batch_size', 16))
    img_size = int(request.form.get('img_size', 640))
    
    # Save and extract dataset
    dataset_file = request.files['dataset']
    if dataset_file and dataset_file.filename.endswith('.zip'):
        filename = secure_filename(dataset_file.filename)
        filepath = os.path.join(app.config['DATASETS_FOLDER'], filename)
        dataset_file.save(filepath)
        
        # Extract dataset
        dataset_path = extract_dataset(filepath)
        
        # Start training
        training_thread = start_training(dataset_path, epochs, model_type, batch_size, img_size)
        
        return jsonify({
            'status': 'success',
            'message': 'Training started',
            'dataset_path': dataset_path
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Invalid dataset file. Must be a ZIP file.'
        }), 400

@app.route('/api/training-status')
def api_training_status():
    """API endpoint to get training status"""
    global training_status
    return jsonify(training_status)

@app.route('/api/start-stream', methods=['POST'])
def api_start_stream():
    """API endpoint to start video stream"""
    global camera, camera_active
    
    # Stop existing stream if any
    if camera_active:
        camera_active = False
        time.sleep(1)  # Give time for the camera to release
        if camera is not None:
            camera.release()
    
    # Start new camera
    camera_id = int(request.form.get('camera_id', 0))
    camera = cv2.VideoCapture(camera_id)
    
    if not camera.isOpened():
        return jsonify({
            'status': 'error',
            'message': f'Failed to open camera {camera_id}'
        }), 500
    
    camera_active = True
    
    return jsonify({
        'status': 'success',
        'message': 'Stream started'
    })

@app.route('/api/stop-stream', methods=['POST'])
def api_stop_stream():
    """API endpoint to stop video stream"""
    global camera, camera_active
    
    camera_active = False
    time.sleep(1)  # Give time for the camera to release
    
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({
        'status': 'success',
        'message': 'Stream stopped'
    })

@app.route('/video-feed')
def video_feed():
    """Video feed endpoint"""
    model_type = request.args.get('model_type', 'detection')
    confidence = float(request.args.get('confidence', 0.25))
    
    return Response(
        get_video_stream(model_type, confidence),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/capture-frame', methods=['POST'])
def api_capture_frame():
    """API endpoint to capture a frame from video stream"""
    global global_frame
    
    if global_frame is None:
        return jsonify({
            'status': 'error',
            'message': 'No frame available'
        }), 400
    
    # Save frame as image
    frame_filename = f"{generate_unique_filename('capture')}.jpg"
    frame_path = os.path.join(app.config['RESULTS_FOLDER'], frame_filename)
    cv2.imwrite(frame_path, global_frame)
    
    return jsonify({
        'status': 'success',
        'frame_url': url_for('results', filename=frame_filename)
    })

@app.route('/results/<filename>')
def results(filename):
    """Serve result files"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/models/<filename>')
def models(filename):
    """Serve model files"""
    return send_file(os.path.join(app.config['MODELS_FOLDER'], filename))

@app.route('/api/list-models')
def api_list_models():
    """API endpoint to list available models"""
    models_list = []
    
    for file in os.listdir(app.config['MODELS_FOLDER']):
        if file.endswith('.pt') or file.endswith('.onnx'):
            models_list.append({
                'name': file,
                'path': url_for('models', filename=file),
                'size': os.path.getsize(os.path.join(app.config['MODELS_FOLDER'], file)),
                'date': os.path.getmtime(os.path.join(app.config['MODELS_FOLDER'], file))
            })
    
    return jsonify(models_list)

if __name__ == '__main__':
    # Crear directorios necesarios si no existen
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    
    # Iniciar la aplicación en el puerto 7860 para compatibilidad con Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)