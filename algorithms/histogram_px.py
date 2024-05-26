import cv2
import matplotlib.pyplot as plt
import numpy as np

def histogram_black_pixels(img_path, threshold=40):
    # Cargar la imagen
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    # Máscara para detectar píxeles oscuros (bajo un umbral)
    mask = np.all(img < [threshold, threshold, threshold], axis=2)
    
    # Contar los píxeles negros y no negros
    black_pixels = np.sum(mask)
    non_black_pixels = img.shape[0] * img.shape[1] - black_pixels
    
    # Crear histograma
    labels = ['Black Pixels', 'Non-Black Pixels']
    counts = [black_pixels, non_black_pixels]
    
    plt.bar(labels, counts, color=['black', 'gray'])
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of Black Pixels')
    plt.show()

# Uso de la función
histogram_black_pixels("C:/Malaria/Dataset/train/cells_train/396.jpg")
