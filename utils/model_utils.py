#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades para extraer información de modelos YOLO.

Este módulo proporciona funciones para extraer metadatos, clases y otra información
relevante de modelos YOLO entrenados, soportando diferentes estructuras de modelos
generados por distintas versiones o herramientas.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from . import yaml_utils

# Configurar logging
logger = logging.getLogger(__name__)

def get_model_basic_info(model_path):
    """
    Obtiene información básica sobre un modelo (tamaño, fecha de creación).
    
    Args:
        model_path (str): Ruta al archivo del modelo (.pt)
        
    Returns:
        dict: Diccionario con información básica del modelo o None si hay error
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Archivo de modelo no encontrado: {model_path}")
            return None
            
        stats = os.stat(model_path)
        
        return {
            'model_size': stats.st_size,
            'created': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"Error obteniendo información básica del modelo {model_path}: {str(e)}")
        return None

def extract_model_metadata(model_dir, model_id):
    """
    Extrae metadatos de un modelo desde diversas fuentes posibles.
    
    Args:
        model_dir (str): Directorio base del modelo
        model_id (str): Identificador único del modelo
        
    Returns:
        dict: Metadatos del modelo combinados de todas las fuentes
    """
    metadata = {}
    
    # Lista de posibles archivos y directorios para metadatos
    metadata_sources = [
        ('model_info.json', os.path.join(model_dir, model_id, 'model_info.json')),
        ('args.yaml', os.path.join(model_dir, model_id, 'args.yaml')),
        ('model_config.json', os.path.join(model_dir, model_id, 'model_config.json')),
        ('data.yaml', os.path.join(model_dir, model_id, 'data.yaml')),
        ('config.yaml', os.path.join(model_dir, model_id, 'config.yaml'))
    ]
    
    # Intentar cargar cada archivo de metadatos
    for source_name, source_path in metadata_sources:
        if os.path.exists(source_path):
            logger.info(f"Encontrada fuente de metadatos: {source_name} en {source_path}")
            
            try:
                if source_name.endswith('.json'):
                    with open(source_path, 'r', encoding='utf-8') as f:
                        source_data = json.load(f)
                        logger.info(f"Metadatos JSON cargados desde {source_name}")
                elif source_name.endswith('.yaml'):
                    source_data = yaml_utils.read_yaml_safe(source_path)
                    logger.info(f"Metadatos YAML cargados desde {source_name}")
                else:
                    continue
                
                # Combinar los metadatos de esta fuente
                if isinstance(source_data, dict):
                    metadata.update(source_data)
            except Exception as e:
                logger.error(f"Error cargando metadatos desde {source_path}: {str(e)}")
    
    return metadata

def extract_class_names(model_path, model_dir, model_id):
    """
    Extrae nombres de clases del modelo YOLO, intentando varios métodos.
    
    Args:
        model_path (str): Ruta al archivo del modelo (.pt)
        model_dir (str): Directorio base de modelos
        model_id (str): Identificador único del modelo
        
    Returns:
        dict: Diccionario de clases en formato {id: nombre_clase}
        o dict vacío si no se encuentran clases
    """
    classes = {}
    
    # Lista de estrategias para extraer clases ordenadas por prioridad
    extraction_strategies = [
        ('modelo_yolo', lambda: _extract_classes_from_yolo_model(model_path)),
        ('data_yaml', lambda: _extract_classes_from_yaml_files(model_dir, model_id)),
        ('results_json', lambda: _extract_classes_from_results(model_dir, model_id))
    ]
    
    # Intentar cada estrategia hasta encontrar clases
    for strategy_name, strategy_func in extraction_strategies:
        try:
            logger.info(f"Intentando extraer clases con estrategia: {strategy_name}")
            strategy_classes = strategy_func()
            
            if strategy_classes and len(strategy_classes) > 0:
                logger.info(f"Clases encontradas usando estrategia {strategy_name}: {strategy_classes}")
                classes = strategy_classes
                break
        except Exception as e:
            logger.error(f"Error en estrategia {strategy_name}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Si no se encontraron clases, usar un valor predeterminado
    if not classes:
        logger.warning(f"No se encontraron clases para el modelo {model_id}, usando valores predeterminados")
        classes = {0: "class0", 1: "class1"}
    
    return classes

def _extract_classes_from_yolo_model(model_path):
    """
    Extrae clases directamente del modelo YOLO cargándolo.
    Esta es la estrategia más precisa pero requiere cargar el modelo.
    
    Args:
        model_path (str): Ruta al archivo del modelo (.pt)
        
    Returns:
        dict: Diccionario de clases o None si hay error
    """
    if not os.path.exists(model_path):
        logger.warning(f"No se puede cargar el modelo, archivo no encontrado: {model_path}")
        return None
        
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Verificar múltiples ubicaciones donde pueden estar las clases
        classes_dict = None
        
        # 1. Verificar model.names (ubicación más común)
        if hasattr(model, 'names') and model.names:
            logger.info(f"Nombres de clases encontrados en model.names: {model.names}")
            classes_dict = model.names
            
        # 2. Verificar model.model.names
        elif hasattr(model, 'model') and hasattr(model.model, 'names') and model.model.names:
            logger.info(f"Nombres de clases encontrados en model.model.names: {model.model.names}")
            classes_dict = model.model.names
            
        # 3. Verificar model.predictor.names
        elif hasattr(model, 'predictor') and hasattr(model.predictor, 'names') and model.predictor.names:
            logger.info(f"Nombres de clases encontrados en model.predictor.names: {model.predictor.names}")
            classes_dict = model.predictor.names
            
        # Normalizar formato (convertir lista a diccionario si es necesario)
        if classes_dict is not None:
            if isinstance(classes_dict, list):
                return {i: name for i, name in enumerate(classes_dict)}
            elif isinstance(classes_dict, dict):
                # Normalizar claves a enteros si es posible
                normalized = {}
                for k, v in classes_dict.items():
                    try:
                        key = int(k) if isinstance(k, str) and k.isdigit() else k
                        normalized[key] = v
                    except (ValueError, TypeError):
                        normalized[k] = v
                return normalized
        
        logger.warning("No se pudieron encontrar clases en el modelo cargado.")
        return None
    except Exception as e:
        logger.error(f"Error cargando modelo para extraer clases: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def _extract_classes_from_yaml_files(model_dir, model_id):
    """
    Extrae clases de archivos YAML relacionados con el modelo.
    
    Args:
        model_dir (str): Directorio base de modelos
        model_id (str): Identificador del modelo
        
    Returns:
        dict: Diccionario de clases o None si no se encuentran
    """
    # Lista de posibles ubicaciones de archivos YAML con información de clases
    yaml_paths = [
        os.path.join(model_dir, model_id, 'data.yaml'),
        os.path.join(model_dir, model_id, 'args.yaml'),
        os.path.join(model_dir, model_id, 'config.yaml'),
        os.path.join('static', 'datasets', 'data', 'data.yaml'),
        os.path.join('static', 'datasets', model_id, 'data.yaml')
    ]
    
    for yaml_path in yaml_paths:
        if os.path.exists(yaml_path):
            logger.info(f"Intentando leer clases desde archivo YAML: {yaml_path}")
            yaml_data = yaml_utils.read_yaml_safe(yaml_path)
            
            if yaml_data:
                classes = yaml_utils.extract_classes_from_yaml(yaml_data)
                if classes:
                    logger.info(f"Clases extraídas de {yaml_path}: {classes}")
                    return classes
    
    logger.warning("No se encontraron clases en archivos YAML")
    return None

def _extract_classes_from_results(model_dir, model_id):
    """
    Extrae clases de archivos de resultados del entrenamiento.
    
    Args:
        model_dir (str): Directorio base de modelos
        model_id (str): Identificador del modelo
        
    Returns:
        dict: Diccionario de clases o None si no se encuentran
    """
    # Posibles archivos de resultados
    results_paths = [
        os.path.join(model_dir, model_id, 'results.json'),
        os.path.join(model_dir, model_id, 'metrics.json'),
        os.path.join(model_dir, model_id, 'training_results.json')
    ]
    
    for result_path in results_paths:
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Buscar clases en varias estructuras posibles
                if 'names' in results:
                    return results['names']
                elif 'classes' in results:
                    return results['classes']
                elif 'model_info' in results and 'classes' in results['model_info']:
                    return results['model_info']['classes']
                
                # Buscar en secciones de métricas por clase
                for key in results:
                    if key.startswith('class_') and isinstance(results[key], dict):
                        class_ids = set()
                        for class_key in results[key]:
                            try:
                                class_id = int(class_key)
                                class_ids.add(class_id)
                            except (ValueError, TypeError):
                                pass
                        
                        if class_ids:
                            return {class_id: f"class{class_id}" for class_id in sorted(class_ids)}
            except Exception as e:
                logger.error(f"Error leyendo archivo de resultados {result_path}: {str(e)}")
    
    # Buscar en directorios de resultados
    results_dir = os.path.join(model_dir, model_id, 'results')
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        for file_name in os.listdir(results_dir):
            if file_name.endswith('.json'):
                try:
                    with open(os.path.join(results_dir, file_name), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        if 'names' in data:
                            return data['names']
                        elif 'classes' in data:
                            return data['classes']
                except Exception as e:
                    logger.error(f"Error leyendo archivo en directorio de resultados: {str(e)}")
    
    logger.warning("No se encontraron clases en archivos de resultados")
    return None
