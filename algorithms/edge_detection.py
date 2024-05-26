import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def image_to_tensor_and_visualize(image_path, is_train=False):
    img = Image.open(image_path).convert('RGB')
    # Convertir la imagen PIL a un array de NumPy para usar OpenCV
    img_np = np.array(img)
    # Convertir de RGB (PIL/OpenCV tiene BGR por defecto) a escala de grises
    gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Aplicar detección de bordes
    edges = cv2.Canny(gray_image, 100, 200)
    # Añadir el canal de bordes a la imagen original
    edges_expanded = np.expand_dims(edges, axis=2)
    img_with_edges = np.concatenate((img_np, edges_expanded), axis=2)
    
    # Visualizar la imagen original y la detección de bordes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original RGB')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Detección de Bordes')
    plt.axis('off')

    # Aunque img_with_edges es una imagen de 4 canales, solo visualizamos los 3 canales RGB aquí
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_edges[:, :, :3])
    plt.title('RGB con Canal de Bordes (Visualización RGB)')
    plt.axis('off')

    plt.show()

# Ejemplo de uso
image_path = 'C:/Users/GeFe/OneDrive/Desktop/TesisMalaria/Dataset/test/41120.jpg'
image_to_tensor_and_visualize(image_path)
