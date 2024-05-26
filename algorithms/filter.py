import cv2
import numpy as np

def apply_contrast(input_image, alpha):
    f_image = input_image.astype(np.float32)
    f_image = np.clip((alpha * (f_image - 128)) + 128, 0, 255)
    return f_image.astype(np.uint8)

def apply_saturation(input_image, saturation_scale):
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[..., 1] = hsv_image[..., 1] * saturation_scale
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_blur(input_image, blur_strength):
    if blur_strength == 0:
        return input_image
    return cv2.GaussianBlur(input_image, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

def apply_sharpen(input_image, strength):
    if strength == 0:
        return input_image
    kernel = np.array([[-1, -1, -1], [-1, 9 + strength, -1], [-1, -1, -1]])
    return cv2.filter2D(input_image, -1, kernel)

image_path = 'images\PLATELET.jpg'  # Reemplaza con la ruta a tu imagen
image = cv2.imread(image_path)

alpha = 1.5  # Contraste
beta = -83    # Exposure
saturation_scale = 2.0  # Saturation
blur_strength = 0  # Blur
sharpen_strength = 0  # Sharpen

adjusted = apply_contrast(image, alpha)
adjusted = cv2.convertScaleAbs(adjusted, alpha=1, beta=beta)
adjusted = apply_saturation(adjusted, saturation_scale)
adjusted = apply_blur(adjusted, blur_strength)
adjusted = apply_sharpen(adjusted, sharpen_strength)

cv2.imshow('Kawaii Result', adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()