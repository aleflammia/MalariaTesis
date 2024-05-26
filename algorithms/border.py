import os
from PIL import Image

# Definir umbral para considerar un píxel como "negro"
BLACK_THRESHOLD = 50  # Los valores RGB menores a este umbral son considerados negros.
PERCENTAGE_THRESHOLD = 0.10  # Porcentaje de píxeles negros requeridos en un borde

def is_pixel_black(pixel):
    r, g, b = pixel[:3]  # Solo considera los tres primeros canales (ignora alpha si existe)
    return r < BLACK_THRESHOLD and g < BLACK_THRESHOLD and b < BLACK_THRESHOLD

def has_black_border(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
       
        # Verificar borde superior
        black_pixels_top = sum(is_pixel_black(img.getpixel((x, 0))) for x in range(width))
        if black_pixels_top/width > PERCENTAGE_THRESHOLD:
            return True
       
        # Verificar borde inferior
        black_pixels_bottom = sum(is_pixel_black(img.getpixel((x, height - 1))) for x in range(width))
        if black_pixels_bottom/width > PERCENTAGE_THRESHOLD:
            return True
       
        # Verificar borde izquierdo
        black_pixels_left = sum(is_pixel_black(img.getpixel((0, y))) for y in range(height))
        if black_pixels_left/height > PERCENTAGE_THRESHOLD:
            return True
       
        # Verificar borde derecho
        black_pixels_right = sum(is_pixel_black(img.getpixel((width - 1, y))) for y in range(height))
        if black_pixels_right/height > PERCENTAGE_THRESHOLD:
            return True

    return False

# Directorio con las imágenes
path = "C:/Malaria/Dataset/train/cells_train"
images_with_black_borders = []

# Verifica si la carpeta existe
if not os.path.exists(path):
    raise ValueError(f"El directorio '{path}' no existe.")

# Recorre las imágenes en la carpeta
for image_file in os.listdir(path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(path, image_file)
        if has_black_border(image_path):
            images_with_black_borders.append(image_file)

n = 0

print("Imágenes con bordes negros:")
for img in images_with_black_borders:
    print(img)
    n+=1

print(n)