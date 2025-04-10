FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear directorio para guardar modelos y uploads si no existen
RUN mkdir -p static/uploads static/results static/models

# Exponer el puerto que utiliza la aplicación
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
