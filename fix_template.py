"""
Script para corregir la estructura de bloques Jinja en training.html
"""

def fix_template():
    # Leer el archivo original
    with open('templates/training.html', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Cortar el archivo en las secciones principales
    parts = content.split('{% block ')
    
    # Crear un nuevo contenido con la estructura correcta
    new_content = parts[0]  # Encabezado
    
    # Procesar cada bloque
    for i in range(1, len(parts)):
        part = parts[i]
        block_name = part.split('%}', 1)[0]
        block_content = part.split('%}', 1)[1]
        
        # Dividir en el próximo endblock, si existe
        if '{% endblock %}' in block_content:
            block_parts = block_content.split('{% endblock %}', 1)
            block_body = block_parts[0]
            remaining = block_parts[1] if len(block_parts) > 1 else ''
            
            # Añadir el bloque con su cierre correcto
            new_content += f'{{% block {block_name}%}}{block_body}{{% endblock %}}\n\n'
            
            # Añadir el resto del contenido (fuera de bloques)
            if remaining:
                new_content += remaining
        else:
            # Si no hay endblock, asumir que el contenido debe estar en este bloque
            new_content += f'{{% block {block_name}%}}{block_content}{{% endblock %}}\n\n'
    
    # Asegurarse de que el contenido del modal esté fuera de los bloques
    if '<div class="modal fade" id="modelArchitectureModal"' in new_content:
        # Mover el modal entre el bloque de contenido y el bloque de JavaScript
        modal_start = new_content.find('<div class="modal fade" id="modelArchitectureModal"')
        modal_end = new_content.find('</div>\n</div>\n</div>', modal_start) + 18
        
        modal_content = new_content[modal_start:modal_end]
        
        # Eliminar el modal de su posición actual
        new_content = new_content[:modal_start] + new_content[modal_end:]
        
        # Insertar el modal después del bloque de contenido
        content_end = new_content.find('{% endblock %}', new_content.find('{% block content %}'))
        new_content = new_content[:content_end+14] + '\n\n' + modal_content + '\n\n' + new_content[content_end+14:]
    
    # Guardar la versión corregida
    with open('templates/training_fixed.html', 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print("Archivo corregido guardado como templates/training_fixed.html")

if __name__ == "__main__":
    fix_template()
