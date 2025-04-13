/**
 * YOLO Model Details Page JavaScript
 * Handles fetching and displaying detailed information about trained YOLO models
 */

// Function to load model details based on model ID
function loadModelDetails(modelId) {
    console.log(`[DEBUG] Loading model details for model ID: ${modelId}`);
    
    // Show loading message in classesStats
    const classesStats = document.getElementById('classesStats');
    if (classesStats) {
        classesStats.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p>Cargando estadísticas...</p></div>';
    } else {
        console.error('[ERROR] Element with ID "classesStats" not found in the DOM');
    }
    
    // Fetch model details from API
    console.log(`[DEBUG] Fetching from API: /api/model-details?id=${encodeURIComponent(modelId)}`);
    fetch(`/api/model-details?id=${encodeURIComponent(modelId)}`)
        .then(response => {
            console.log(`[DEBUG] API Response status:`, response.status);
            if (!response.ok) {
                throw new Error('Model not found or error fetching model details');
            }
            return response.json();
        })
        .then(data => {
            console.log('[DEBUG] Model data received from API:', data);
            renderModelDetails(data);
        })
        .catch(error => {
            console.error('[ERROR] Error fetching model details:', error);
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
    console.log('[DEBUG] Rendering model details:', model);
    
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
    
    // Download links
    updateDownloadLinks(model);
    
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
    
    // Get metrics data only once
    const metrics = model.metrics || {};
    console.log('[DEBUG] Metrics data:', metrics);
    
    // Performance metrics
    const metricsDiv = document.getElementById('performanceMetrics');
    metricsDiv.innerHTML = '';
    
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
    
    // Add debug information
    if (model.model_info || model.classes) {
        const debugInfo = document.getElementById('debugInfo');
        const debugData = document.getElementById('debugData');
        
        // Format debug data in a readable way
        if (debugInfo && debugData) {
            const debugObj = {
                model_info: model.model_info || {},
                metrics: metrics,
                has_classes: model.model_info && model.model_info.classes ? 
                    Object.keys(model.model_info.classes).length : 0
            };
            debugData.textContent = JSON.stringify(debugObj, null, 2);
            debugInfo.classList.remove('d-none'); // Make debug info visible during development
        }
    }
    
    // -----------------------------------------------
    // Class Statistics - SIMPLIFIED APPROACH
    // -----------------------------------------------
    const classesStats = document.getElementById('classesStats');
    if (classesStats) {
        // First check if we have classes data in the model
        let classes = null;
        
        if (model.model_info && model.model_info.classes) {
            classes = model.model_info.classes;
        } else if (model.classes) {
            classes = model.classes;  
        }
        
        if (classes) {
            const classCount = Object.keys(classes).length;
            
            // Build simple HTML structure similar to other sections
            let html = `
                <div class="mb-3">
                    <div class="d-flex justify-content-between mb-2">
                        <span class="fw-bold">Total Classes:</span>
                        <span class="badge bg-primary">${classCount}</span>
                    </div>`;
                    
            // Add precision/recall if available        
            if (metrics.precision || metrics['metrics/precision(B)']) {
                html += `
                    <div class="d-flex justify-content-between mb-2">
                        <span class="fw-bold">Average Precision:</span>
                        <span class="badge bg-success">${(metrics.precision || metrics['metrics/precision(B)']).toFixed(2)}</span>
                    </div>`;
            }
            
            if (metrics.recall || metrics['metrics/recall(B)']) {
                html += `
                    <div class="d-flex justify-content-between mb-2">
                        <span class="fw-bold">Average Recall:</span>
                        <span class="badge bg-info">${(metrics.recall || metrics['metrics/recall(B)']).toFixed(2)}</span>
                    </div>`;
            }
            
            html += `</div>`;
                
            // Add class distribution
            html += `
                <div class="mt-3">
                    <h6 class="mb-2">Classes</h6>
                    <div class="d-flex flex-wrap small">`;
                    
            Object.entries(classes).forEach(([id, name]) => {
                html += `
                    <div class="me-2 mb-1">
                        <span class="badge bg-secondary">${id}</span> ${name}
                    </div>`;
            });
            
            html += `
                    </div>
                </div>`;
                
            classesStats.innerHTML = html;
        } else {
            // No classes data available
            classesStats.innerHTML = `
                <div class="alert alert-info">
                    <p><i class="bi bi-info-circle"></i> No class data available for this model.</p>
                </div>`;
        }
    }
    
    // Classes section
    const classesContainer = document.getElementById('classesContainer');
    classesContainer.innerHTML = '';
    
    // Asegurarnos de tener datos de clases, buscando en todas las posibles ubicaciones
    let classesData = null;
    
    if (model.classes && Object.keys(model.classes).length > 0) {
        classesData = model.classes;
        console.log("[DEBUG] Using direct model.classes data");
    } else if (model.model_info && model.model_info.names && Object.keys(model.model_info.names).length > 0) {
        classesData = model.model_info.names;
        console.log("[DEBUG] Using model_info.names data");
    }
    
    console.log("[DEBUG] Final classes data:", classesData);
    
    // Convert classesData to standard format if it's an array
    if (Array.isArray(classesData)) {
        console.log("Converting array classes to object format");
        const tempClassesData = {};
        classesData.forEach((className, index) => {
            tempClassesData[index] = className;
        });
        classesData = tempClassesData;
    }
    
    // Ultima verificación: usar clases COCO por defecto si no hay clases definidas
    if (!classesData || Object.keys(classesData).length === 0) {
        console.log("No se encontraron clases. Intentando extraer clases de otras partes del modelo...");
        
        let foundClasses = false;
        // Buscar en outputs - algunos modelos YOLO almacenan las clases aquí
        if (model && model.outputs) {
            const outputKeys = Object.keys(model.outputs);
            outputKeys.forEach(key => {
                if (model.outputs[key].classes && Object.keys(model.outputs[key].classes).length > 0) {
                    console.log(`Encontradas clases en model.outputs[${key}].classes`);
                    classesData = model.outputs[key].classes;
                    foundClasses = true;
                }
            });
        }
        
        // Buscar en el modelo directamente para modelos YOLO custom
        if (!foundClasses && model && model.names) {
            console.log("Encontradas clases en model.names");
            classesData = model.names;
            foundClasses = true;
        }
        
        // Buscar en la respuesta del modelo para entrenamientos recientes
        if (!foundClasses && model && model.task === 'detect' && !classesData) {
            console.log("Intentando crear clases básicas para modelo de detección");
            // Para modelos de detección, podemos usar clases genéricas numbered
            classesData = {};
            // Empezamos con 2 clases genéricas y luego ajustamos
            const genericClassCount = 2;
            for (let i = 0; i < genericClassCount; i++) {
                classesData[i] = `class${i}`;
            }
            foundClasses = true;
            
            console.log("Creadas clases básicas:", classesData);
        }
        
        // Si encontramos clases, mostramos las tarjetas
        if (foundClasses && Object.keys(classesData).length > 0) {
            console.log("Mostrando clases encontradas en fuentes alternativas:", classesData);
            
            // Mostrar mensaje de introducción
            const classCount = Object.keys(classesData).length;
            const classIntro = document.createElement('div');
            classIntro.classList.add('class-intro', 'mb-4');
            classIntro.innerHTML = `<p>Este modelo puede detectar ${classCount} clases diferentes. Cada tarjeta a continuación representa una clase.</p>`;
            classesContainer.appendChild(classIntro);
            
            // Crear tarjetas para cada clase
            const classCardsRow = document.createElement('div');
            classCardsRow.classList.add('row', 'row-cols-1', 'row-cols-md-2', 'row-cols-lg-3', 'g-4');
            
            // Ordenar clases por ID
            const sortedClassIds = Object.keys(classesData).sort((a, b) => {
                return parseInt(a) - parseInt(b);
            });
            
            // Crear tarjetas para las clases
            sortedClassIds.forEach(classId => {
                const className = classesData[classId];
                
                const cardCol = document.createElement('div');
                cardCol.classList.add('col');
                
                const card = document.createElement('div');
                card.classList.add('card', 'h-100', 'class-card');
                
                const cardBody = document.createElement('div');
                cardBody.classList.add('card-body');
                
                const cardTitle = document.createElement('h5');
                cardTitle.classList.add('card-title');
                cardTitle.textContent = className;
                
                const cardText = document.createElement('div');
                cardText.classList.add('card-text');
                cardText.innerHTML = `
                    <span class="badge bg-primary mb-2">ID: ${classId}</span>
                    <p>Esta clase representa objetos de tipo <strong>${className}</strong> que el modelo puede identificar.</p>
                `;
                
                cardBody.appendChild(cardTitle);
                cardBody.appendChild(cardText);
                card.appendChild(cardBody);
                cardCol.appendChild(card);
                classCardsRow.appendChild(cardCol);
            });
            
            classesContainer.appendChild(classCardsRow);
        } else {
            // Fallback para cuando no hay información de clases
            classesContainer.innerHTML = `
                <div class="col-12">
                    <div class="alert alert-info">
                        <h5><i class="bi bi-info-circle"></i> Class Information</h5>
                        <p>This model has been trained for object detection. However, detailed class information is not available.</p>
                        <p>The model may still be able to detect common objects like persons, cars, animals, etc. depending on the training dataset.</p>
                    </div>
                </div>
            `;
        }
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
    // Anteriormente, esto solo configuraba los botones con ID específicos
    // document.getElementById('downloadPyTorch').href = model.path;
    
    // Ahora seleccionamos todos los enlaces que contienen "Download PyTorch" en su texto
    const ptDownloadButtons = document.querySelectorAll('a.btn:not([id])[download]:has(i.bi-download)');
    
    ptDownloadButtons.forEach(button => {
        if (button.textContent.includes('PyTorch')) {
            button.href = model.path;
        } else if (button.textContent.includes('ONNX')) {
            if (model.has_onnx) {
                button.href = model.onnx_path;
                button.closest('.card')?.style.display = 'block';
            } else {
                button.closest('.card')?.style.display = 'none';
            }
        }
    });
    
    // Mantener compatibilidad con el código anterior para los botones con IDs
    const downloadPyTorch = document.getElementById('downloadPyTorch');
    if (downloadPyTorch) {
        downloadPyTorch.href = model.path;
    }
    
    // Check if ONNX model is available
    const onnxCard = document.getElementById('onnxCard');
    const downloadONNX = document.getElementById('downloadONNX');
    
    if (model.has_onnx) {
        if (downloadONNX) {
            downloadONNX.href = model.onnx_path;
        }
        if (onnxCard) {
            onnxCard.style.display = 'block';
        }
    } else {
        if (onnxCard) {
            onnxCard.style.display = 'none';
        }
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

// Function to update download links
function updateDownloadLinks(model) {
    // Anteriormente, esto solo configuraba los botones con ID específicos
    // document.getElementById('downloadPyTorch').href = model.path;
    
    // Ahora seleccionamos todos los enlaces que contienen "Download PyTorch" en su texto
    const ptDownloadButtons = document.querySelectorAll('a.btn:not([id])[download]:has(i.bi-download)');
    
    ptDownloadButtons.forEach(button => {
        if (button.textContent.includes('PyTorch')) {
            button.href = model.path;
        } else if (button.textContent.includes('ONNX')) {
            if (model.has_onnx) {
                button.href = model.onnx_path;
                button.closest('.card')?.style.display = 'block';
            } else {
                button.closest('.card')?.style.display = 'none';
            }
        }
    });
    
    // Mantener compatibilidad con el código anterior para los botones con IDs
    const downloadPyTorch = document.getElementById('downloadPyTorch');
    if (downloadPyTorch) {
        downloadPyTorch.href = model.path;
    }
}
