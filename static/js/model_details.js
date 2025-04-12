/**
 * YOLO Model Details Page JavaScript
 * Handles fetching and displaying detailed information about trained YOLO models
 */

// Function to load model details based on model ID
function loadModelDetails(modelId) {
    // Fetch model details from API
    fetch(`/api/model-details?id=${encodeURIComponent(modelId)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Model not found or error fetching model details');
            }
            return response.json();
        })
        .then(data => {
            // Render model details to the page
            renderModelDetails(data);
        })
        .catch(error => {
            console.error('Error fetching model details:', error);
            // Show error message on the page
            document.querySelector('.container').innerHTML = `
                <div class="alert alert-danger">
                    <h4><i class="bi bi-exclamation-triangle"></i> Error Loading Model</h4>
                    <p>${error.message}</p>
                    <a href="/training" class="btn btn-outline-danger mt-3">Back to Training</a>
                </div>
            `;
        });
}

// Function to render model details to the page
function renderModelDetails(model) {
    // Update page title and header
    document.title = `${model.name} - YOLO Vision AI`;
    document.getElementById('modelName').textContent = model.name;
    document.getElementById('modelDescription').textContent = `Trained on ${model.created} with ${model.model_info.classes ? Object.keys(model.model_info.classes).length : 0} classes`;
    
    // Update badges
    document.getElementById('modelType').textContent = model.model_info.base_model || 'YOLOv8';
    document.getElementById('modelSize').textContent = formatFileSize(model.model_info.model_size);
    document.getElementById('modelTask').textContent = 'Object Detection';
    
    // Update model overview
    let overview = 'This YOLO model has been trained for object detection tasks';
    if (model.model_info.classes) {
        const classCount = Object.keys(model.model_info.classes).length;
        overview += ` and can detect ${classCount} different ${classCount === 1 ? 'class' : 'classes'} of objects.`;
    } else {
        overview += '.';
    }
    document.getElementById('modelOverview').textContent = overview;
    
    // Model specifications
    const specsTable = document.getElementById('modelSpecsTable');
    specsTable.innerHTML = '';
    
    const specs = [
        { name: 'Architecture', value: 'YOLO (You Only Look Once)' },
        { name: 'Version', value: 'YOLOv8' },
        { name: 'Base Model', value: model.model_info.base_model || 'YOLOv8n' },
        { name: 'Type', value: model.type || 'PyTorch' },
        { name: 'Task', value: 'Object Detection' },
        { name: 'Input Resolution', value: `${model.model_info.imgsz || 640}×${model.model_info.imgsz || 640}` },
        { name: 'File Size', value: formatFileSize(model.model_info.model_size) },
        { name: 'Created', value: model.created },
        { name: 'Framework', value: 'Ultralytics' }
    ];
    
    specs.forEach(spec => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <th scope="row">${spec.name}</th>
            <td>${spec.value}</td>
        `;
        specsTable.appendChild(row);
    });
    
    // Training configuration
    const trainingConfigTable = document.getElementById('trainingConfigTable');
    trainingConfigTable.innerHTML = '';
    
    const trainingParams = [
        { name: 'Epochs', value: model.model_info.epochs || 'N/A' },
        { name: 'Batch Size', value: model.model_info.batch || 'N/A' },
        { name: 'Learning Rate', value: model.model_info.lr0 || 'N/A' },
        { name: 'Image Size', value: model.model_info.imgsz || 640 },
        { name: 'Optimizer', value: model.model_info.optimizer || 'SGD' },
        { name: 'Dataset', value: model.model_info.dataset || 'Custom Dataset' }
    ];
    
    trainingParams.forEach(param => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <th scope="row">${param.name}</th>
            <td>${param.value}</td>
        `;
        trainingConfigTable.appendChild(row);
    });
    
    // Performance metrics
    const metricsDiv = document.getElementById('performanceMetrics');
    metricsDiv.innerHTML = '';
    
    const metrics = model.metrics || {};
    
    const metricItems = [
        { name: 'Precision', value: (metrics.precision || metrics['metrics/precision(B)'] || 0).toFixed(4), color: 'success' },
        { name: 'Recall', value: (metrics.recall || metrics['metrics/recall(B)'] || 0).toFixed(4), color: 'info' },
        { name: 'mAP50', value: (metrics.mAP50 || metrics['metrics/mAP50(B)'] || 0).toFixed(4), color: 'primary' },
        { name: 'mAP50-95', value: (metrics.mAP50_95 || metrics['metrics/mAP50-95(B)'] || 0).toFixed(4), color: 'secondary' }
    ];
    
    metricItems.forEach(metric => {
        const metricDiv = document.createElement('div');
        metricDiv.className = 'metrics-badge';
        metricDiv.innerHTML = `
            <span class="badge bg-${metric.color} p-2" style="width: 100%; text-align: left;">
                ${metric.name}: ${metric.value}
            </span>
        `;
        metricsDiv.appendChild(metricDiv);
    });
    
    // Classes section
    const classesContainer = document.getElementById('classesContainer');
    classesContainer.innerHTML = '';
    
    if (model.model_info.classes) {
        const classes = model.model_info.classes;
        Object.entries(classes).forEach(([classId, className]) => {
            const classCol = document.createElement('div');
            classCol.className = 'col-lg-3 col-md-4 col-sm-6 mb-3';
            classCol.innerHTML = `
                <div class="card h-100">
                    <div class="card-body text-center">
                        <span class="badge bg-dark mb-2">${className}</span>
                        <p class="card-text small text-muted mb-0">Class ID: ${classId}</p>
                    </div>
                </div>
            `;
            classesContainer.appendChild(classCol);
        });
    } else {
        classesContainer.innerHTML = `
            <div class="col-12">
                <div class="alert alert-warning">
                    <i class="bi bi-exclamation-triangle"></i> No class information available for this model.
                </div>
            </div>
        `;
    }
    
    // Model structure visualization
    const modelStructure = document.getElementById('modelStructure');
    modelStructure.innerHTML = `
        <div class="alert alert-info">
            <p><strong>YOLOv8 Architecture:</strong> The model consists of a backbone (feature extraction), a neck (feature aggregation), and a head (final predictions).</p>
            <p><strong>Backbone:</strong> CSPDarknet - Extracts features from the input image using convolutional layers.</p>
            <p><strong>Neck:</strong> PANet (Path Aggregation Network) - Enhances feature fusion across different scales.</p>
            <p><strong>Head:</strong> Decoupled detection head - Produces the final outputs (object classes, bounding boxes).</p>
        </div>
    `;
    
    // Set download links
    document.getElementById('downloadPyTorch').href = model.path;
    
    // Check if ONNX model is available
    const onnxCard = document.getElementById('onnxCard');
    if (model.has_onnx) {
        document.getElementById('downloadONNX').href = model.onnx_path;
        onnxCard.style.display = 'block';
    } else {
        onnxCard.style.display = 'none';
    }
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
