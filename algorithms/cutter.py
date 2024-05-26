import cv2
import numpy as np
import os

def recortar_celula(imagen_path, salida_path):
    # Leer la imagen
    imagen = cv2.imread(imagen_path, cv2.IMREAD_COLOR)
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Binarización
    _, binarizada = cv2.threshold(gris, 1, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encerrar los contornos encontrados en un rectángulo
    x_min, y_min, w, h = cv2.boundingRect(np.vstack(contornos))

    # Asegurarse de que las dimensiones son enteros
    x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
    
    # Ajustar al tamaño máximo para que sea un cuadrado
    maximo_lado = max(w, h)
    x_centro, y_centro = x_min + w // 2, y_min + h // 2
    x_min = x_centro - maximo_lado // 2
    y_min = y_centro - maximo_lado // 2

    # Recortar la imagen usando las dimensiones del rectángulo
    recorte = imagen[y_min:y_min+maximo_lado, x_min:x_min+maximo_lado]
    
    cv2.imwrite(salida_path, recorte)

# Ruta a la carpeta que contiene las imágenes

directorio_imagenes = "C:/Malaria/Dataset/train/cells_train/"

def es_imagen_valida(nombre):
    # Verificar si el archivo tiene extensión .jpg
    if not nombre.endswith('.jpg'):
        return False
    # Verificar si el nombre (sin la extensión) es numérico
    nombre_sin_extension = nombre.split('.')[0]
    return nombre_sin_extension.isdigit()

# Filtrar los nombres de archivo válidos
nombres_validos = filter(es_imagen_valida, os.listdir(directorio_imagenes))

# Ordenar los nombres numéricamente
nombres_ordenados = sorted(nombres_validos, key=lambda x: int(x.split(".")[0]))

for nombre_imagen in nombres_ordenados:
    ruta_imagen = os.path.join(directorio_imagenes, nombre_imagen)
    
    # Verificar si es una imagen y si sus dimensiones son 140x140px
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
    
    if imagen is not None and imagen.shape[0] == 140 and imagen.shape[1] == 140:
        recortar_celula(ruta_imagen, ruta_imagen)  # Sobreescribir la imagen original