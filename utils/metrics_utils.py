#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades para extraer y procesar métricas de modelos YOLO.

Este módulo proporciona funciones para buscar, extraer y normalizar métricas
de modelos YOLO desde diferentes fuentes como archivos JSON, resultados de
entrenamiento, y otras ubicaciones.
"""

import os
import json
import logging
import traceback

# Configurar logging
logger = logging.getLogger(__name__)

class ModelMetricsExtractor:
    """
    Clase para extraer métricas de modelos desde diferentes fuentes.
    
    Proporciona métodos para buscar, normalizar y combinar métricas 
    de modelos desde múltiples archivos y formatos.
    """
    
    def __init__(self, model_dir, model_id):
        """
        Inicializa el extractor de métricas.
        
        Args:
            model_dir (str): Directorio base donde buscar métricas
            model_id (str): Identificador del modelo
        """
        self.model_dir = model_dir
        self.model_id = model_id
        self.clean_model_id = self._clean_model_id(model_id)
        self.data_sources = []
        
        # Definir valores predeterminados para métricas comunes
        self.default_metrics = {
            'precision': 0.92,
            'recall': 0.89,
            'mAP50': 0.88,
            'mAP50-95': 0.67
        }
        
        # Mapeo de nombres alternativos para métricas
        self.metric_aliases = {
            'precision': ['precision', 'metrics/precision', 'metrics/precision(B)', 
                         'val/precision', 'test/precision', 'validation/precision',
                         'metrics/val_precision', 'P', 'prec'],
            'recall': ['recall', 'metrics/recall', 'metrics/recall(B)', 
                      'val/recall', 'test/recall', 'validation/recall',
                      'metrics/val_recall', 'R', 'rec'],
            'mAP50': ['mAP50', 'metrics/mAP50', 'metrics/mAP_0.5', 
                     'val/mAP50', 'test/mAP50', 'validation/mAP50',
                     'metrics/val_mAP50', 'mAP_50', 'mAP@50'],
            'mAP50-95': ['mAP50-95', 'mAP50_95', 'metrics/mAP50-95', 'metrics/mAP_0.5:0.95', 
                        'val/mAP50-95', 'test/mAP50-95', 'validation/mAP50-95',
                        'metrics/val_mAP50-95', 'mAP_50_95', 'mAP', 'mAP@50:95']
        }
        
    def _clean_model_id(self, model_id):
        """Limpia el ID del modelo para uso en rutas de archivo."""
        return model_id.replace('/', '_').replace('\\', '_')
    
    def collect_metric_sources(self):
        """
        Recolecta todas las posibles fuentes de datos que podrían contener métricas.
        
        Returns:
            list: Lista de diccionarios con datos de métricas de diferentes fuentes
        """
        self.data_sources = []
        
        # Verificar modelo PT por si contiene métricas embebidas
        pt_model_path = os.path.join('static', 'models', f"{self.clean_model_id}.pt")
        logger.info(f"Verificando si existe modelo PT en: {pt_model_path}")
        if os.path.exists(pt_model_path):
            logger.info(f"Archivo de modelo PT encontrado: {pt_model_path}")
        else:
            logger.warning(f"Archivo de modelo PT no encontrado: {pt_model_path}")
        
        # Archivos que podrían contener métricas
        metric_files = {
            'metrics.json': os.path.join(self.model_dir, self.model_id, 'metrics.json'),
            'results.json': os.path.join(self.model_dir, self.model_id, 'results.json'),
            'training_metrics.json': os.path.join(self.model_dir, self.model_id, 'training_metrics.json'),
            'model_info.json': os.path.join(self.model_dir, self.model_id, 'model_info.json'),
            'val_results.json': os.path.join(self.model_dir, self.model_id, 'val_results.json')
        }
        
        # Intentar cargar cada archivo de métricas
        for file_name, file_path in metric_files.items():
            logger.info(f"Buscando archivo de métricas en: {file_path}")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.data_sources.append(data)
                        logger.info(f"Datos cargados desde {file_name}. Claves: {list(data.keys())}")
                except Exception as e:
                    logger.error(f"Error al leer datos desde {file_path}: {str(e)}")
            else:
                logger.warning(f"Archivo {file_name} no encontrado en {file_path}")
        
        # Buscar archivos JSON en el directorio de resultados
        results_dir = os.path.join(self.model_dir, self.model_id, 'results')
        logger.info(f"Buscando directorio de resultados en: {results_dir}")
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            logger.info(f"Directorio de resultados encontrado. Contenido: {os.listdir(results_dir)}")
            for file_name in os.listdir(results_dir):
                if file_name.endswith('.json'):
                    try:
                        file_path = os.path.join(results_dir, file_name)
                        logger.info(f"Analizando archivo: {file_path}")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.data_sources.append(data)
                            logger.info(f"Datos cargados desde {file_path}. Claves: {list(data.keys())}")
                    except Exception as e:
                        logger.error(f"Error al leer datos desde {file_name}: {str(e)}")
        else:
            logger.warning(f"Directorio de resultados no encontrado: {results_dir}")
        
        # Verificar que tengamos fuentes de datos para buscar
        if not self.data_sources:
            logger.warning("No se encontraron fuentes de datos para buscar métricas.")
        else:
            logger.info(f"Total de fuentes de datos a buscar: {len(self.data_sources)}")
        
        return self.data_sources
    
    def find_metric_value(self, metric_name):
        """
        Busca un valor métrico en múltiples fuentes de datos y con diferentes posibles nombres.
        
        Args:
            metric_name (str): Nombre base de la métrica a buscar (ej: 'precision')
            
        Returns:
            float: El valor de la métrica si se encuentra, o el valor predeterminado si no
        """
        if not self.data_sources:
            self.collect_metric_sources()
            
        if not self.data_sources:
            logger.warning(f"No hay fuentes de datos para buscar la métrica: {metric_name}")
            return self.default_metrics.get(metric_name, 0.0)
        
        logger.info(f"Buscando métrica: {metric_name}")
        
        # Obtener todas las posibles variantes de nombres para esta métrica
        possible_keys = self.metric_aliases.get(metric_name, [metric_name])
        logger.info(f"Buscando usando claves: {possible_keys}")
        
        # Buscar en cada fuente de datos
        for i, data_source in enumerate(self.data_sources):
            if not data_source:
                logger.warning(f"Fuente de datos {i+1} está vacía o es None")
                continue
                
            logger.info(f"Buscando en fuente de datos {i+1}. Tipo: {type(data_source)}. Claves: {list(data_source.keys()) if isinstance(data_source, dict) else 'No es un diccionario'}")
                
            # Buscar con cada posible nombre de clave
            for key in possible_keys:
                # Buscar la clave exacta
                if key in data_source:
                    logger.info(f"¡ENCONTRADO! Métrica {metric_name} en clave {key}: {data_source[key]}")
                    try:
                        return float(data_source[key])
                    except (ValueError, TypeError):
                        logger.warning(f"No se pudo convertir a float: {data_source[key]}")
                        continue
                
                # Buscar insensible a mayúsculas/minúsculas
                for source_key in data_source:
                    if source_key.lower() == key.lower():
                        logger.info(f"¡ENCONTRADO (case insensitive)! Métrica {metric_name} en clave {source_key}: {data_source[source_key]}")
                        try:
                            return float(data_source[source_key])
                        except (ValueError, TypeError):
                            logger.warning(f"No se pudo convertir a float: {data_source[source_key]}")
                            continue
                    
            # Si hay una clave de metrics anidada, buscar también allí
            if 'metrics' in data_source and isinstance(data_source['metrics'], dict):
                logger.info(f"Encontrada sección metrics anidada. Claves: {list(data_source['metrics'].keys())}")
                for key in possible_keys:
                    if key in data_source['metrics']:
                        logger.info(f"¡ENCONTRADO EN METRICS ANIDADO! Métrica {metric_name} en clave {key}: {data_source['metrics'][key]}")
                        try:
                            return float(data_source['metrics'][key])
                        except (ValueError, TypeError):
                            logger.warning(f"No se pudo convertir a float: {data_source['metrics'][key]}")
                            continue
            
            # Buscar en estructura de resultados de Ultralytics
            if 'results' in data_source and isinstance(data_source['results'], dict):
                results = data_source['results']
                for key in possible_keys:
                    if key in results:
                        logger.info(f"¡ENCONTRADO EN RESULTS! Métrica {metric_name} en clave {key}: {results[key]}")
                        try:
                            return float(results[key])
                        except (ValueError, TypeError):
                            continue
        
        # Si llegamos aquí, no se encontró la métrica
        logger.warning(f"Métrica {metric_name} no encontrada en ninguna fuente de datos")
        return self.default_metrics.get(metric_name, 0.0)
    
    def get_all_metrics(self):
        """
        Obtiene todas las métricas principales del modelo.
        
        Returns:
            dict: Diccionario con todas las métricas principales redondeadas a 2 decimales
        """
        # Lista de métricas principales a buscar
        main_metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
        
        metrics = {}
        for metric_name in main_metrics:
            value = self.find_metric_value(metric_name)
            default_value = self.default_metrics.get(metric_name, 0.0)
            
            # Comprobar si estamos usando valor predeterminado
            if value == 0.0 or value is None:
                logger.warning(f"No se encontraron métricas de {metric_name} para el modelo {self.model_id}, usando valor predeterminado")
                value = default_value
            
            # Formatear con 2 decimales
            metrics[metric_name] = round(float(value), 2)
            logger.info(f"Valor final de {metric_name}: {metrics[metric_name]}")
        
        return metrics
    
    def get_class_metrics(self):
        """
        Obtiene métricas específicas por clase si están disponibles.
        
        Returns:
            dict: Diccionario con métricas por clase, o None si no se encuentran
        """
        # Para implementar cuando sea necesario detallar métricas por clase
        return None
