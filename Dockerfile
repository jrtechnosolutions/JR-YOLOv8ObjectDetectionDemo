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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear directorio para guardar modelos y uploads si no existen
RUN mkdir -p static/uploads static/results static/models

# Exponer el puerto que utiliza la aplicación
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
