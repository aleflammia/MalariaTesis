import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Directorio con las imágenes
path = "c:/Malaria/Dataset/train/cells_train"

# Verifica si la carpeta existe
if not os.path.exists(path):
    raise ValueError(f"El directorio '{path}' no existe.")

# Lista para guardar las dimensiones de las imágenes
dimensions = []

# Recorre las imágenes en la carpeta
for image_file in os.listdir(path):
    # Solo procesa archivos con extensiones de imágenes comunes
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        with Image.open(os.path.join(path, image_file)) as img:
            width, height = img.size
            dimensions.append((width, height))

# Contar las dimensiones repetidas
counter = Counter(dimensions)

# Ordenar por dimensiones (primero por ancho, luego por alto)
sorted_dimensions = sorted(counter.items(), key=lambda x: (x[0][0], x[0][1]))

# Preparar datos para el histograma
labels, values = zip(*sorted_dimensions)
labels_str = [f"{dim[0]}x{dim[1]}" for dim in labels]

# Graficar histograma
plt.figure(figsize=(15, 7))
bars = plt.bar(labels_str, values)

# Añadir la cantidad de fotos sobre cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)),
             ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=90)
plt.ylabel('Cantidad de Imágenes')
plt.xlabel('Dimensiones (AnchoxAlto)')
plt.title('Histograma de Dimensiones de Imágenes en la Carpeta "cells"')
plt.tight_layout()
plt.show()
