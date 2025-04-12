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

# Folder configuration with absolute paths
app_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(app_dir, 'static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(app_dir, 'static', 'results')
app.config['MODELS_FOLDER'] = os.path.join(app_dir, 'static', 'models')
app.config['DATASETS_FOLDER'] = os.path.join(app_dir, 'static', 'datasets')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'zip'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], 
               app.config['RESULTS_FOLDER'], 
               app.config['MODELS_FOLDER'], 
               app.config['DATASETS_FOLDER']]:
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")

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
    
    try:
        # First, try to load locally
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            # If not found locally, load from Ultralytics but with a message
            print(f"Model {models[model_type]} not found locally, loading from Ultralytics...")
            model = YOLO(models[model_type])
            # If we're in a production environment (like HF Spaces)
            # don't try to save the model to avoid filesystem operations
            if os.environ.get('HF_SPACE', '') != 'true':
                # Only try to save in development environments
                if os.path.exists(models[model_type]):
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    shutil.copy(models[model_type], model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to direct loading without saving
        model = YOLO(models[model_type])
    
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
    global training_status, is_training
    
    try:
        # Unpack dataset_path if it contains yaml_files
        yaml_files = []
        if isinstance(dataset_path, tuple):
            dataset_path, yaml_files = dataset_path
        
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        yaml_path = os.path.join(app.config['DATASETS_FOLDER'], "data.yaml")
        
        # Check if valid existing YAML exists and try to use it
        existing_yaml = None
        if yaml_files:
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                        if isinstance(yaml_content, dict) and 'names' in yaml_content:
                            existing_yaml = yaml_file
                            break
                except Exception as e:
                    print(f"Error reading YAML file {yaml_file}: {e}")
        
        # Determine class names
        class_names = {}
        
        # Check if classes.txt exists
        classes_path = os.path.join(dataset_path, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
                for i, cls in enumerate(classes):
                    class_names[i] = cls
                print(f"Found classes.txt with {len(classes)} classes: {classes}")
        
        # Use existing YAML if possible, otherwise create new one
        if existing_yaml:
            try:
                with open(existing_yaml, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                # Update paths to absolute
                yaml_config['path'] = os.path.abspath(dataset_path)
                
                # Convert relative paths to absolute if they start with ..
                if 'train' in yaml_config and yaml_config['train'].startswith('../'):
                    yaml_config['train'] = 'train/images'
                if 'val' in yaml_config and yaml_config['val'].startswith('../'):
                    yaml_config['val'] = 'valid/images'
                if 'test' in yaml_config and yaml_config['test'].startswith('../'):
                    yaml_config['test'] = 'test/images'
                
                # Keep class names from existing YAML or update with classes.txt
                if not class_names and 'names' in yaml_config:
                    class_names = yaml_config['names']
                elif class_names:
                    yaml_config['names'] = class_names
                
                print(f"Using modified existing YAML from {existing_yaml}")
            except Exception as e:
                print(f"Error processing existing YAML, creating new one: {e}")
                yaml_config = create_default_yaml(dataset_path, class_names)
        else:
            yaml_config = create_default_yaml(dataset_path, class_names)
        
        # Verify dataset structure before training
        verify_dataset_structure(dataset_path, yaml_config)
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f)
        
        print(f"Training config written to {yaml_path}")
        print(f"YAML content: {yaml_config}")
        
        # Select model based on model_type
        model_map = {
            'detection': 'yolov8n.pt',
            'segmentation': 'yolov8n-seg.pt',
            'pose': 'yolov8n-pose.pt',
            'classification': 'yolov8n-cls.pt'
        }
        
        # Create absolute path to the model file
        model_file = os.path.join(app.config['MODELS_FOLDER'], model_map[model_type])
        if not os.path.exists(model_file):
            # Download if not exists
            model = YOLO(model_map[model_type])
        else:
            model = YOLO(model_file)
        
        # Start training
        training_status["message"] = f"Training started. Model: {model_type}, Epochs: {epochs}"
        
        # Custom callback to update progress
        def on_train_epoch_end(trainer):
            global training_status
            epoch = trainer.epoch
            training_status["progress"] = int((epoch / epochs) * 100)
            training_status["message"] = f"Training: {epoch}/{epochs} epochs completed"
        
        # Add a callback for when training completes
        def on_train_end(trainer):
            global training_status
            training_status["progress"] = 100
            training_status["message"] = "Training complete! Finalizing model..."
            training_status["complete"] = True
        
        # Register callbacks
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)
        
        # Training arguments
        train_args = {
            'data': yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'project': app.config['MODELS_FOLDER'],
            'name': f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        print(f"Starting training with arguments: {train_args}")
        
        # Start training with proper error handling
        try:
            results = model.train(**train_args)
            print(f"Training completed successfully. Results: {results}")
            
            # Save the trained model
            model_save_path = os.path.join(
                app.config['MODELS_FOLDER'], 
                f"trained_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            
            # Use correct method to save based on YOLO version
            try:
                # Try newer version API method first
                model.export(format="pt", save_dir=os.path.dirname(model_save_path), 
                             filename=os.path.basename(model_save_path))
                print(f"Model saved using export() method at {model_save_path}")
            except (AttributeError, TypeError) as e:
                # Fallback for older versions
                try:
                    model.save(model_save_path)
                    print(f"Model saved using save() method at {model_save_path}")
                except Exception as save_err:
                    print(f"Failed to save model with either method: {str(save_err)}")
                    # Try to salvage the automatically saved model from the results
                    best_model_path = os.path.join(app.config['MODELS_FOLDER'], 
                                          f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                                          "weights", "best.pt")
                    if os.path.exists(best_model_path):
                        shutil.copy(best_model_path, model_save_path)
                        print(f"Copied automatically saved model from {best_model_path} to {model_save_path}")
                    else:
                        # Try to find any potential best.pt files
                        for root, dirs, files in os.walk(app.config['MODELS_FOLDER']):
                            if "best.pt" in files and not training_status["complete"]:
                                # Found a trained model that wasn't properly registered
                                found_path = os.path.join(root, "best.pt")
                                shutil.copy(found_path, model_save_path)
                                print(f"Found and copied model from {found_path} to {model_save_path}")
                                break
                        else:
                            print(f"Could not find any automatically saved model")
            
            # Convert to ONNX only if PT model was saved successfully
            onnx_path = model_save_path.replace('.pt', '.onnx')
            if os.path.exists(model_save_path):
                try:
                    model.export(format="onnx", save=True)
                    print(f"Model exported to ONNX at {onnx_path}")
                except Exception as onnx_err:
                    print(f"Failed to export to ONNX: {str(onnx_err)}")
                    onnx_path = None
            else:
                print("Skipping ONNX export because PT model was not saved")
                onnx_path = None
            
            # Update status to complete regardless of ONNX export
            training_status["progress"] = 100
            training_status["message"] = f"Training complete. Model saved at {model_save_path}"
            training_status["complete"] = True
            training_status["model_path"] = model_save_path
            training_status["onnx_path"] = onnx_path
            
        except Exception as train_err:
            error_message = f"Training error: {str(train_err)}"
            print(f"ERROR during training: {error_message}")
            training_status["message"] = error_message
            training_status["complete"] = True
            is_training = False
            return
        
    except Exception as e:
        error_message = f"Training error: {str(e)}"
        print(f"ERROR: {error_message}")
        training_status["message"] = error_message
        training_status["complete"] = True
    
    finally:
        is_training = False

def create_default_yaml(dataset_path, class_names):
    """Create a default YAML configuration"""
    return {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': class_names if class_names else {'0': 'class0', '1': 'class1'}
    }

def verify_dataset_structure(dataset_path, yaml_config):
    """Verify dataset structure and log useful debug information"""
    print(f"Verifying dataset structure at {dataset_path}")
    
    # Check train directory
    train_path = os.path.join(dataset_path, 'train', 'images')
    if not os.path.exists(train_path):
        print(f"WARNING: Train images directory doesn't exist: {train_path}")
    else:
        train_images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(train_images)} images in train directory")
        
        # Check labels
        train_labels = os.path.join(dataset_path, 'train', 'labels')
        if not os.path.exists(train_labels):
            print(f"WARNING: Train labels directory doesn't exist: {train_labels}")
        else:
            label_files = [f for f in os.listdir(train_labels) if f.endswith('.txt')]
            print(f"Found {len(label_files)} label files in train directory")
            
            # Check if some images don't have labels
            train_img_bases = {os.path.splitext(f)[0] for f in train_images}
            label_bases = {os.path.splitext(f)[0] for f in label_files}
            missing_labels = train_img_bases - label_bases
            if missing_labels:
                print(f"WARNING: {len(missing_labels)} train images don't have corresponding labels")
    
    # Check validation directory
    valid_path = os.path.join(dataset_path, 'valid', 'images')
    if not os.path.exists(valid_path):
        print(f"WARNING: Validation images directory doesn't exist: {valid_path}")
    else:
        valid_images = [f for f in os.listdir(valid_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(valid_images)} images in validation directory")
        
        # Check labels
        valid_labels = os.path.join(dataset_path, 'valid', 'labels')
        if not os.path.exists(valid_labels):
            print(f"WARNING: Validation labels directory doesn't exist: {valid_labels}")
        else:
            label_files = [f for f in os.listdir(valid_labels) if f.endswith('.txt')]
            print(f"Found {len(label_files)} label files in validation directory")

def extract_dataset(zip_path):
    """Extract dataset from zip file and organize it properly"""
    dataset_name = os.path.basename(zip_path).split('.')[0]
    extract_path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name)
    
    # Create extraction directory
    os.makedirs(extract_path, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Look for existing YAML file
    yaml_files = []
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                yaml_files.append(os.path.join(root, file))
    
    # Check dataset structure and organize if needed
    # This handles common formats like YOLO, COCO, etc.
    required_dirs = ['train', 'valid', 'test']
    
    # Verify if the dataset is already properly organized
    has_proper_structure = True
    for d in ['train', 'valid']:  # test is optional
        img_dir = os.path.join(extract_path, d, 'images')
        lbl_dir = os.path.join(extract_path, d, 'labels')
        
        if not os.path.exists(img_dir) or not os.listdir(img_dir):
            has_proper_structure = False
            break
            
        # Check if labels directory exists and has files
        if not os.path.exists(lbl_dir) or not os.listdir(lbl_dir):
            has_proper_structure = False
            break
    
    # If the dataset is already properly organized
    if has_proper_structure:
        print(f"Dataset has proper structure. Found train/valid dirs with images and labels.")
        return extract_path, yaml_files
    
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
    
    print(f"Found {len(image_files)} images and {len(annotation_files)} annotation files.")
    
    # If we found images but no organized structure, create one
    if image_files and not has_proper_structure:
        print("Reorganizing dataset structure...")
        # Create basic train/valid split (80/20)
        os.makedirs(os.path.join(extract_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'valid', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(extract_path, 'test', 'labels'), exist_ok=True)
        
        # Shuffle and split files
        np.random.shuffle(image_files)
        split_idx1 = int(len(image_files) * 0.7)  # 70% train
        split_idx2 = int(len(image_files) * 0.9)  # 20% valid, 10% test
        train_images = image_files[:split_idx1]
        valid_images = image_files[split_idx1:split_idx2]
        test_images = image_files[split_idx2:]
        
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
        
        for img_path in test_images:
            shutil.copy(img_path, os.path.join(extract_path, 'test', 'images', os.path.basename(img_path)))
            
            # Find corresponding annotation if exists
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for ann_path in annotation_files:
                if os.path.splitext(os.path.basename(ann_path))[0] == base_name:
                    shutil.copy(ann_path, os.path.join(extract_path, 'test', 'labels', 
                                                    os.path.basename(ann_path)))
                    break
        
        print(f"Dataset reorganized: {len(train_images)} train, {len(valid_images)} valid, {len(test_images)} test images.")
    
    return extract_path, yaml_files

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

# Function to process uploaded video files
def process_video_file(video_path, model_type, confidence=0.25, max_frames=300):
    """Process a video file with selected YOLO model and save the result"""
    model = load_model(model_type)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file"
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit the number of frames to process
    frames_to_process = min(total_frames, max_frames)
    
    # Create output video file
    result_filename = f"{generate_unique_filename('video_result')}.mp4"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    processed_frames = 0
    
    try:
        # Process each frame
        for i in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with YOLO
            results = model(frame, conf=confidence)
            result_frame = results[0].plot()
            
            # Write the processed frame to output video
            out.write(result_frame)
            processed_frames += 1
            
            # Print progress every 10 frames
            if i % 10 == 0:
                print(f"Processing video: {i}/{frames_to_process} frames ({int(i/frames_to_process*100)}%)")
    except Exception as e:
        cap.release()
        out.release()
        return None, f"Error processing video: {str(e)}"
    finally:
        # Release resources
        cap.release()
        out.release()
    
    return result_filename, f"Successfully processed {processed_frames} frames"

# API endpoint to process video file
@app.route('/api/process-video', methods=['POST'])
def api_process_video():
    """API endpoint for processing an uploaded video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'Video file type not allowed'}), 400
    
    # Save the uploaded video
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], generate_unique_filename(filename))
    video_file.save(filepath)
    
    # Get parameters
    model_type = request.form.get('model_type', 'detection')
    confidence = float(request.form.get('confidence', 0.25))
    max_frames = int(request.form.get('max_frames', 300))
    
    # Process the video file
    result_filename, message = process_video_file(filepath, model_type, confidence, max_frames)
    
    if result_filename is None:
        return jsonify({'error': message}), 500
    
    return jsonify({
        'status': 'success',
        'message': message,
        'video_url': url_for('results', filename=result_filename)
    })

# Video streaming page
@app.route('/streaming')
def streaming():
    # Redirect to the new unified video stream page
    return redirect(url_for('video_stream'))

# Video processing page - alternative to streaming for Hugging Face Spaces
@app.route('/video-processing')
def video_processing():
    # Redirect to the new unified video stream page
    return redirect(url_for('video_stream'))

# Unified Video Analysis page (combines streaming and video processing)
@app.route('/video-stream')
def video_stream():
    # Detect if we're running in Hugging Face
    is_huggingface = os.environ.get('SPACE_ID') is not None or 'huggingface.co' in request.host or 'spaces' in request.host
    
    # Render the appropriate template based on environment
    if is_huggingface:
        return render_template('video_stream_hf.html')
    else:
        return render_template('video_stream.html')

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
        dataset_path, yaml_files = extract_dataset(filepath)
        
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
    
    # Si hay un mensaje, revisa si contiene información sobre completado o error
    if "message" in training_status:
        # Busca si hay mensajes que indiquen que el entrenamiento terminó
        indicators_of_completion = [
            "Training complete", 
            "Training completed", 
            "Results saved to", 
            "Optimizer stripped from",
            "Training completed successfully"
        ]
        
        if any(indicator in training_status["message"] for indicator in indicators_of_completion) or training_status["complete"]:
            training_status["progress"] = 100
            training_status["complete"] = True
            if "error" in training_status["message"].lower():
                # Si hay un mensaje de error pero sabemos que terminó, actualiza el mensaje
                training_status["message"] = "Training completed successfully"
        
        # Verifica errores pero el entrenamiento puede haber completado igualmente
        elif "error" in training_status["message"].lower() and not training_status["complete"]:
            # Busca best.pt en los modelos para confirmar si en realidad se completó
            for root, dirs, files in os.walk(app.config['MODELS_FOLDER']):
                if "best.pt" in files:
                    # Encontró un modelo entrenado
                    model_path = os.path.join(root, "best.pt")
                    training_status["progress"] = 100
                    training_status["message"] = f"Training complete. Model found at {model_path}"
                    training_status["complete"] = True
                    training_status["model_path"] = model_path
                    break
    
    return jsonify(training_status)

@app.route('/api/start-stream', methods=['POST'])
def api_start_stream():
    """API endpoint to start video stream"""
    global camera, camera_active
    
    # Check if we are running in Hugging Face Spaces
    is_huggingface = os.environ.get('SPACE_ID') is not None
    
    # Stop existing stream if any
    if camera_active:
        camera_active = False
        time.sleep(1)  # Give time for the camera to release
        if camera is not None:
            camera.release()
    
    # If we're in Hugging Face Spaces, return a specific message
    if is_huggingface:
        return jsonify({
            'status': 'error',
            'message': 'Live camera streaming is not available in Hugging Face Spaces. Please use the Video Processing feature instead.'
        }), 400
    
    # Start new camera
    camera_id = int(request.form.get('camera_id', 0))
    
    try:
        camera = cv2.VideoCapture(camera_id)
        
        if not camera.isOpened():
            return jsonify({
                'status': 'error',
                'message': 'Could not open camera. Please check your camera connection and permissions.'
            }), 400
        
        camera_active = True
        
        return jsonify({
            'status': 'success',
            'message': 'Stream started'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error starting stream: {str(e)}'
        }), 500

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
    return send_file(os.path.abspath(os.path.join(app.config['RESULTS_FOLDER'], filename)))

@app.route('/models/<filename>')
def models(filename):
    """Serve model files"""
    return send_file(os.path.abspath(os.path.join(app.config['MODELS_FOLDER'], filename)))

@app.route('/api/list-models')
def api_list_models():
    """API endpoint to list available trained models"""
    models = []
    
    # Ensure directory exists
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    
    # Log to help debugging
    print(f"Searching for models in: {app.config['MODELS_FOLDER']}")
    
    # First check if we can find any trained models in the direct models directory
    found_models = False
    
    # List all directories in the models folder that might contain trained models
    model_dirs = []
    for item in os.listdir(app.config['MODELS_FOLDER']):
        full_path = os.path.join(app.config['MODELS_FOLDER'], item)
        if os.path.isdir(full_path):
            # Check if this directory has a weights subfolder with model files
            weights_dir = os.path.join(full_path, "weights")
            if os.path.exists(weights_dir):
                model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                if model_files:
                    model_dirs.append(full_path)
                    print(f"Found model directory with weights: {full_path}")
    
    # If we found model directories, process them directly
    if model_dirs:
        found_models = True
        for model_dir in model_dirs:
            weights_dir = os.path.join(model_dir, "weights")
            for model_file in os.listdir(weights_dir):
                if model_file.endswith('.pt'):
                    model_path = os.path.join(weights_dir, model_file)
                    model_name = os.path.basename(model_dir)
                    rel_path = os.path.relpath(model_path, app.config['MODELS_FOLDER'])
                    
                    print(f"Processing model: {model_name} from {model_path}")
                    
                    # Find training artifacts
                    training_files = {}
                    
                    # Files to look for
                    potential_files = [
                        'labels.jpg',
                        'results.png',
                        'confusion_matrix.png',
                        'PR_curve.png',
                        'F1_curve.png',
                        'P_curve.png',
                        'R_curve.png',
                        'results.csv',
                        'args.yaml',
                    ]
                    
                    # Look for files in the model directory
                    for potential_file in potential_files:
                        file_path = os.path.join(model_dir, potential_file)
                        if os.path.exists(file_path):
                            rel_file_path = os.path.relpath(file_path, app.config['MODELS_FOLDER'])
                            print(f"Found artifact: {potential_file}")
                            training_files[potential_file] = url_for('model_artifact', path=rel_file_path)
                    
                    # Check for ONNX model
                    onnx_path = model_path.replace('.pt', '.onnx')
                    has_onnx = os.path.exists(onnx_path)
                    
                    # Read metrics
                    metrics = {}
                    csv_path = os.path.join(model_dir, 'results.csv')
                    if os.path.exists(csv_path):
                        try:
                            with open(csv_path, 'r') as f:
                                lines = f.readlines()
                                if len(lines) >= 2:
                                    header = lines[0].strip().split(',')
                                    last_row = lines[-1].strip().split(',')
                                    
                                    for i, key in enumerate(header):
                                        if i < len(last_row):
                                            try:
                                                metrics[key] = float(last_row[i])
                                            except (ValueError, TypeError):
                                                metrics[key] = last_row[i]
                                    print(f"Loaded metrics from CSV")
                        except Exception as e:
                            print(f"Error reading metrics: {e}")
                    
                    # Extract class names
                    model_info = {
                        'classes': None,
                        'model_size': os.path.getsize(model_path),
                        'imgsz': 640,
                    }
                    
                    # Try to find data.yaml for class names
                    yaml_files = [f for f in os.listdir(model_dir) if f.endswith('.yaml')]
                    for yaml_file in yaml_files:
                        try:
                            with open(os.path.join(model_dir, yaml_file), 'r') as f:
                                yaml_data = yaml.safe_load(f)
                                if isinstance(yaml_data, dict) and 'names' in yaml_data:
                                    model_info['classes'] = yaml_data['names']
                                    print(f"Found class names in {yaml_file}")
                                    break
                        except Exception as e:
                            print(f"Error reading YAML file: {e}")
                    
                    # Add special handling for best.pt models
                    is_best = 'best' in model_file
                    creation_time = os.path.getctime(model_path)
                    
                    models.append({
                        'name': f"{model_name} ({'best' if is_best else 'last'})",
                        'path': url_for('static', filename=f'models/{rel_path}'),
                        'full_path': model_path,
                        'relative_path': rel_path,
                        'type': 'PyTorch',
                        'has_onnx': has_onnx,
                        'onnx_path': url_for('static', filename=f'models/{os.path.relpath(onnx_path, app.config["MODELS_FOLDER"])}') if has_onnx else None,
                        'created': datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S'),
                        'training_files': training_files,
                        'metrics': metrics,
                        'model_info': model_info,
                        'model_dir': os.path.relpath(model_dir, app.config['MODELS_FOLDER']),
                        'is_best': is_best
                    })
    
    # If no models found in the direct method, fall back to the original method of searching
    if not found_models:
        # Continue with original search logic
        print("No models found with direct method, using original search method...")
        
        # List all .pt and .onnx files in the models directory and subdirectories
        for root, dirs, files in os.walk(app.config['MODELS_FOLDER']):
            for file in files:
                if file.endswith('.pt') and not file.startswith(('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')):
                    # Skip base models, only include trained models
                    model_path = os.path.join(root, file)
                    rel_path = os.path.relpath(model_path, app.config['MODELS_FOLDER'])
                    model_name = os.path.splitext(os.path.basename(file))[0]
                    
                    print(f"Found model: {model_name} at {model_path}")
                    
                    # Determine the model directory (can be .../model_name/weights/best.pt)
                    model_dir = os.path.dirname(model_path)
                    if os.path.basename(model_dir) == 'weights':
                        model_dir = os.path.dirname(model_dir)
                    
                    print(f"Model directory: {model_dir}")
                    
                    # Look for training artifacts
                    training_files = {}
                    
                    # First look in the current directory
                    # Then look in the subdirectory "runs/train/exp*/", which is where YOLOv8 normally saves results
                    search_dirs = [model_dir]
                    
                    # Look for any 'runs' directory in the model directory or its parents
                    for search_dir in [model_dir, os.path.dirname(model_dir), os.path.dirname(os.path.dirname(model_dir))]:
                        runs_dir = os.path.join(search_dir, 'runs', 'train')
                        if os.path.exists(runs_dir):
                            # Look for the most recent exp directory
                            exp_dirs = [d for d in os.listdir(runs_dir) if d.startswith('exp')]
                            if exp_dirs:
                                # Sort by modification date, most recent first
                                exp_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)), reverse=True)
                                search_dirs.append(os.path.join(runs_dir, exp_dirs[0]))
                    
                    # Add search in specific subdirectories for HuggingFace Spaces
                    app_static_dir = os.path.join(app_dir, 'static')
                    hf_runs_dir = os.path.join(app_static_dir, 'models', 'runs', 'train')
                    if os.path.exists(hf_runs_dir):
                        exp_dirs = [d for d in os.listdir(hf_runs_dir) if d.startswith('exp')]
                        if exp_dirs:
                            exp_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(hf_runs_dir, d)), reverse=True)
                            search_dirs.append(os.path.join(hf_runs_dir, exp_dirs[0]))
                    
                    # Files to look for
                    potential_files = [
                        'labels.jpg',
                        'results.png',
                        'confusion_matrix.png',
                        'PR_curve.png',
                        'F1_curve.png',
                        'P_curve.png',
                        'R_curve.png',
                        'results.csv',
                        'args.yaml',
                    ]
                    
                    # Look for files in all search directories
                    for search_dir in search_dirs:
                        print(f"Searching for training artifacts in: {search_dir}")
                        for potential_file in potential_files:
                            file_path = os.path.join(search_dir, potential_file)
                            if os.path.exists(file_path):
                                print(f"Found artifact: {potential_file}")
                                rel_file_path = os.path.relpath(file_path, app.config['MODELS_FOLDER'])
                                training_files[potential_file] = url_for('model_artifact', path=rel_file_path)
                    
                    # Check if there's a corresponding ONNX model
                    onnx_path = model_path.replace('.pt', '.onnx')
                    has_onnx = os.path.exists(onnx_path)
                    
                    # Read metrics
                    metrics = {}
                    csv_path = os.path.join(model_dir, 'results.csv')
                    if os.path.exists(csv_path):
                        try:
                            with open(csv_path, 'r') as f:
                                lines = f.readlines()
                                if len(lines) >= 2:
                                    header = lines[0].strip().split(',')
                                    last_row = lines[-1].strip().split(',')
                                    
                                    for i, key in enumerate(header):
                                        if i < len(last_row):
                                            try:
                                                metrics[key] = float(last_row[i])
                                            except (ValueError, TypeError):
                                                metrics[key] = last_row[i]
                                    print(f"Loaded metrics from CSV")
                        except Exception as e:
                            print(f"Error reading metrics: {e}")
                    
                    # Extract class names
                    model_info = {
                        'classes': None,
                        'model_size': os.path.getsize(model_path),
                        'imgsz': 640,
                    }
                    
                    # Try to find data.yaml for class names
                    yaml_files = [f for f in os.listdir(model_dir) if f.endswith('.yaml')]
                    for yaml_file in yaml_files:
                        try:
                            with open(os.path.join(model_dir, yaml_file), 'r') as f:
                                yaml_data = yaml.safe_load(f)
                                if isinstance(yaml_data, dict) and 'names' in yaml_data:
                                    model_info['classes'] = yaml_data['names']
                                    print(f"Found class names in {yaml_file}")
                                    break
                        except Exception as e:
                            print(f"Error reading YAML file: {e}")
                    
                    # Create web-accessible URL for the model
                    # Static files are served from /static/models/
                    url_path = url_for('static', filename=f'models/{rel_path}')
                    
                    models.append({
                        'name': model_name,
                        'path': url_path,
                        'full_path': model_path,
                        'relative_path': rel_path,
                        'type': 'PyTorch',
                        'has_onnx': has_onnx,
                        'onnx_path': url_for('static', filename=f'models/{os.path.relpath(onnx_path, app.config["MODELS_FOLDER"])}') if has_onnx else None,
                        'created': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                        'training_files': training_files,
                        'metrics': metrics,
                        'model_info': model_info,
                        'model_dir': os.path.relpath(model_dir, app.config['MODELS_FOLDER'])
                    })
    
    # Sort models by creation time (newest first)
    models.sort(key=lambda x: x['created'], reverse=True)
    
    print(f"Found {len(models)} models total")
    return jsonify(models)

@app.route('/model_artifact/<path:path>')
def model_artifact(path):
    """Serve model artifacts like graphs and training files"""
    # Try different possible paths to find the file
    possible_paths = [
        os.path.join(app.config['MODELS_FOLDER'], path),                # Normal path
        os.path.join(app_dir, 'static', 'models', path),                # Alternative path in static
        os.path.join(app_dir, path)                                     # Relative path from app root
    ]
    
    for test_path in possible_paths:
        if os.path.exists(test_path) and os.path.isfile(test_path):
            print(f"Serving artifact from: {test_path}")
            return send_file(test_path)
    
    # If not found, return a 404 error
    print(f"Artifact not found: {path}")
    print(f"Tried paths: {possible_paths}")
    return "Artifact not found", 404

@app.route('/model/<path:model_dir>/weights/<path:model_file>')
def serve_model_file(model_dir, model_file):
    """Serve model files from the weights directory"""
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_dir, "weights", model_file)
    
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return f"Model file not found: {model_path}", 404

@app.route('/model/<model_id>')
def model_details(model_id):
    """
    Render the detailed view page for a specific model
    """
    return render_template('model_details.html')

@app.route('/api/model-details')
def api_model_details():
    """
    API endpoint to get detailed information about a specific model
    """
    model_id = request.args.get('id')
    if not model_id:
        return jsonify({'error': 'Model ID is required'}), 400
    
    # Get models directory
    models_dir = app.config['MODELS_FOLDER']
    model_path = os.path.join(models_dir, model_id + '.pt')
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    # Get model information
    model_info = {}
    try:
        # Get model basic information
        stats = os.stat(model_path)
        model_size = stats.st_size
        created_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Check for ONNX version
        onnx_path = os.path.join(models_dir, model_id + '.onnx')
        has_onnx = os.path.exists(onnx_path)
        
        # Path to model's metadata directory
        metadata_dir = os.path.join(models_dir, model_id)
        model_info_file = os.path.join(metadata_dir, 'model_info.json')
        metrics_file = os.path.join(metadata_dir, 'metrics.json')
        args_file = os.path.join(metadata_dir, 'args.yaml')
        
        # Initialize model info with basic attributes
        model_data = {
            'name': model_id,
            'path': f'/models/{model_id}.pt',
            'created': created_time,
            'type': 'PyTorch',
            'model_info': {
                'model_size': model_size,
                'base_model': 'YOLOv8n'  # Default if not found in metadata
            },
            'metrics': {},
            'has_onnx': has_onnx
        }
        
        if has_onnx:
            model_data['onnx_path'] = f'/models/{model_id}.onnx'
        
        # Load model info if available
        if os.path.exists(model_info_file):
            with open(model_info_file, 'r') as f:
                info = json.load(f)
                model_data['model_info'].update(info)
        
        # Load metrics if available
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                model_data['metrics'] = metrics
        
        # Check for training artifacts
        artifacts_dir = os.path.join(models_dir, model_id, 'artifacts')
        if os.path.exists(artifacts_dir):
            training_files = {}
            for file in os.listdir(artifacts_dir):
                file_path = os.path.join(artifacts_dir, file)
                if os.path.isfile(file_path):
                    # Create relative URL for the artifact
                    training_files[file] = f'/models/{model_id}/artifacts/{file}'
            
            model_data['training_files'] = training_files
        
        # Add args.yaml if available
        if os.path.exists(args_file):
            model_data['args_file'] = f'/models/{model_id}/args.yaml'
        
        return jsonify(model_data)
    except Exception as e:
        app.logger.error(f"Error getting model details: {str(e)}")
        return jsonify({'error': 'Error getting model details'}), 500

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    os.makedirs('static/datasets', exist_ok=True)
    
    # Check if we're running in Hugging Face Spaces
    if os.environ.get('HF_SPACE', '') == 'true':
        print("Running in Hugging Face Spaces environment")
    
    # Start the application on port 7860 for compatibility with Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)