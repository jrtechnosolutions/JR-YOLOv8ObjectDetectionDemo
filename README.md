# YOLO Vision AI

Una aplicación web completa basada en Flask que aprovecha los modelos YOLO para diversas tareas de visión por computadora.

## Características

- **Detección de Objetos**: Detecta objetos en imágenes utilizando YOLOv8
- **Segmentación de Imágenes**: Genera máscaras precisas a nivel de píxel para objetos detectados
- **Estimación de Pose**: Detecta puntos clave del cuerpo humano
- **Clasificación de Imágenes**: Clasifica imágenes en diferentes categorías
- **Entrenamiento de Modelos**: Entrena modelos YOLO personalizados con tus propios datos
- **Transmisión de Video**: Procesa video en tiempo real desde tu webcam

## Tecnologías Utilizadas

- Flask (Backend)
- Bootstrap 5 (Frontend)
- Ultralytics YOLO (Modelos de IA)
- OpenCV (Procesamiento de imágenes y video)
- PyTorch (Framework de aprendizaje profundo)

## Cómo Usar

1. Sube una imagen o proporciona una URL
2. Selecciona el tipo de análisis (detección, segmentación, pose, clasificación)
3. Ajusta el umbral de confianza según sea necesario
4. Visualiza los resultados de manera interactiva

## Entrenamiento de Modelos Personalizados

Puedes entrenar tus propios modelos YOLO subiendo un conjunto de datos en formato ZIP. La aplicación te guiará a través del proceso y te permitirá descargar el modelo entrenado para uso futuro.

## Transmisión de Video en Vivo

Conecta tu webcam para analizar video en tiempo real utilizando cualquiera de los modelos YOLO disponibles.
