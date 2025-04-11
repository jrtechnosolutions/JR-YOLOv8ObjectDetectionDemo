FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para Matplotlib y Ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear directorio para guardar modelos y uploads si no existen y dar permisos adecuados
RUN mkdir -p /app/static/uploads /app/static/results /app/static/models /app/uploads /app/results /app/models \
    && chmod -R 777 /app/static /app/uploads /app/results /app/models

# Exponer el puerto que utiliza la aplicación
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
