#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades para el manejo seguro de archivos YAML.

Este módulo proporciona funciones robustas para leer y procesar archivos
YAML, manejando diferentes formatos y errores comunes que pueden aparecer
en archivos generados por herramientas externas como RoboFlow.
"""

import os
import yaml
import logging

# Configurar logging
logger = logging.getLogger(__name__)

def read_yaml_safe(yaml_path, default_value=None, fix_syntax=True):
    """
    Lee un archivo YAML de manera segura, manejando posibles errores de sintaxis.
    
    Args:
        yaml_path (str): Ruta al archivo YAML a leer
        default_value (any, optional): Valor a devolver si falla la lectura. Por defecto None.
        fix_syntax (bool, optional): Intenta corregir errores comunes de sintaxis. Por defecto True.
        
    Returns:
        dict: Contenido del archivo YAML, o default_value si falla la lectura
    """
    if not os.path.exists(yaml_path):
        logger.warning(f"El archivo YAML no existe: {yaml_path}")
        return default_value
    
    try:
        # Intento estándar de lectura
        with open(yaml_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                logger.info(f"Archivo YAML leído correctamente: {yaml_path}")
                return data
            except yaml.YAMLError as e:
                if not fix_syntax:
                    logger.error(f"Error de sintaxis en YAML: {str(e)}")
                    return default_value
                
                # Si hay error y fix_syntax está activado, intentamos reparar el archivo
                logger.warning(f"Intentando reparar archivo YAML: {yaml_path}")
                return _try_fix_yaml(yaml_path)
    except Exception as e:
        logger.error(f"Error leyendo archivo YAML {yaml_path}: {str(e)}")
        return default_value

def _try_fix_yaml(yaml_path):
    """
    Intenta reparar y leer un archivo YAML con errores de sintaxis comunes.
    
    Args:
        yaml_path (str): Ruta al archivo YAML a reparar
        
    Returns:
        dict: Contenido del archivo YAML reparado, o None si falla la reparación
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detectar y corregir errores comunes:
        
        # 1. Comillas faltantes en valores que contienen espacios
        fixed_content = content
        
        # 2. Errores en la sangría
        fixed_content = _fix_indentation(fixed_content)
        
        # 3. Caracteres de escape incorrectos
        fixed_content = _fix_escape_chars(fixed_content)
        
        # Intenta cargar el contenido reparado
        try:
            data = yaml.safe_load(fixed_content)
            logger.info(f"YAML reparado y leído correctamente: {yaml_path}")
            return data
        except yaml.YAMLError:
            # Si aún falla, intenta un enfoque línea por línea
            return _parse_yaml_line_by_line(content)
    except Exception as e:
        logger.error(f"Error reparando archivo YAML {yaml_path}: {str(e)}")
        return None

def _fix_indentation(content):
    """Intenta corregir problemas de sangría en el contenido YAML."""
    fixed_lines = []
    current_key = None
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            # Si la línea contiene un valor pero no está correctamente indentada
            if ':' in stripped and not stripped.endswith(':'):
                key, value = stripped.split(':', 1)
                fixed_lines.append(f"{key}:{value}")
            else:
                fixed_lines.append(stripped)
        else:
            fixed_lines.append(line)
    return '\n'.join(fixed_lines)

def _fix_escape_chars(content):
    """Corrige caracteres de escape problemáticos."""
    # Reemplaza secuencias de escape comunes que causan problemas
    replacements = {
        '\\\\': '\\',  # Doble backslash
        '\\"': '"',    # Comillas escapadas incorrectamente
        '\\\'': '\'',  # Comillas simples escapadas incorrectamente
    }
    
    fixed_content = content
    for old, new in replacements.items():
        fixed_content = fixed_content.replace(old, new)
    
    return fixed_content

def _parse_yaml_line_by_line(content):
    """
    Intenta analizar un archivo YAML línea por línea para extraer pares clave-valor.
    Este es un último recurso cuando los métodos normales de análisis fallan.
    """
    result = {}
    current_key = None
    list_data = []
    is_in_list = False
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Detectar si estamos dentro de una lista
        if line.startswith('-'):
            is_in_list = True
            item = line[1:].strip()
            if current_key:
                if current_key not in result:
                    result[current_key] = []
                result[current_key].append(item)
            else:
                list_data.append(item)
        elif ':' in line:
            is_in_list = False
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else None
            
            if key and value:
                # Si el valor parece ser un número
                if value.isdigit():
                    result[key] = int(value)
                elif _is_float(value):
                    result[key] = float(value)
                else:
                    # Eliminar comillas si están presentes
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    result[key] = value
            else:
                current_key = key
                result[key] = {}
    
    # Si teníamos datos de lista sin clave, los ponemos bajo 'items'
    if list_data and not is_in_list:
        result['items'] = list_data
    
    logger.info(f"YAML parseado manualmente: se extrajeron {len(result)} claves")
    return result

def _is_float(text):
    """Comprueba si una cadena puede convertirse a float."""
    try:
        float(text)
        return True
    except ValueError:
        return False

def extract_classes_from_yaml(yaml_data):
    """
    Extrae los nombres de las clases desde datos YAML.
    Soporta diferentes formatos (diccionario o lista).
    
    Args:
        yaml_data (dict): Datos YAML ya cargados
        
    Returns:
        dict: Diccionario de clases en formato {id: nombre_clase}
        o None si no se encuentran clases
    """
    if not yaml_data:
        return None
    
    # Buscar clases en la estructura del YAML
    if 'names' in yaml_data:
        class_data = yaml_data['names']
    elif 'classes' in yaml_data:
        class_data = yaml_data['classes']
    else:
        # Buscar en otras posibles ubicaciones
        for key in yaml_data:
            if isinstance(yaml_data[key], dict) and 'names' in yaml_data[key]:
                class_data = yaml_data[key]['names']
                break
        else:
            logger.warning("No se encontraron nombres de clases en los datos YAML")
            return None
    
    # Convertir a formato diccionario si es una lista
    if isinstance(class_data, list):
        return {i: name for i, name in enumerate(class_data)}
    elif isinstance(class_data, dict):
        # Asegurar que las claves son enteros
        normalized_classes = {}
        for key, value in class_data.items():
            try:
                # Intentar convertir la clave a entero si es posible
                normalized_key = int(key) if isinstance(key, str) and key.isdigit() else key
                normalized_classes[normalized_key] = value
            except (ValueError, TypeError):
                normalized_classes[key] = value
        return normalized_classes
    
    logger.warning(f"Formato de clases desconocido: {type(class_data)}")
    return None
