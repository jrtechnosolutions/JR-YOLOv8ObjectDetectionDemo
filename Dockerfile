FROM python:3.10-slim

WORKDIR /app

# Instalar solo dependencias esenciales del sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para Matplotlib y Ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

# Predescargar los modelos más pequeños para evitar la descarga durante la inicialización
RUN mkdir -p /app/static/models

# Primero instalar PyTorch y dependencias críticas
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 flask==2.0.1 
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear todos los directorios necesarios y dar permisos completos
RUN mkdir -p /app/static/uploads /app/static/results /app/static/models /app/static/datasets \
    && chmod -R 777 /app /tmp

# Exponer el puerto que utiliza la aplicación
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
