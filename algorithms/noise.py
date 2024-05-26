import os
import cv2
import numpy as np

def black_pixels_at_edge(mask, edge_width_ratio=0.05):
    height, width = mask.shape[:2]
    edge_width = int(width * edge_width_ratio)
    edge_height = int(height * edge_width_ratio)
    
    left_edge_mask = mask[:, :edge_width]
    right_edge_mask = mask[:, -edge_width:]
    top_edge_mask = mask[:edge_height, :]
    bottom_edge_mask = mask[-edge_height:, :]

    return np.any(left_edge_mask) or np.any(right_edge_mask) or \
           np.any(top_edge_mask) or np.any(bottom_edge_mask)

def replace_black_with_noise_edge_only(img, threshold=40, edge_width_ratio=0.05):
    mask = np.all(img < [threshold, threshold, threshold], axis=2)
    if black_pixels_at_edge(mask, edge_width_ratio):
        mask_uchar = (mask * 255).astype(np.uint8)
        mean, std_dev = cv2.meanStdDev(img, mask=~mask_uchar)
        mean = mean.squeeze()
        std_dev = std_dev.squeeze()

        # Nos aseguramos de que el ruido se genere para cada canal por separado
        noise = np.zeros(img.shape, np.uint8)
        for i in range(3):  # Suponemos img.shape[2] es 3 para RGB
            noise[:, :, i] = np.random.normal(mean[i], std_dev[i], img.shape[:2])

        return np.where(mask[:, :, None], noise, img)
    return img

def process_folder_with_edge_noise(folder_path, output_folder_path, threshold=40, edge_width_ratio=0.05):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        processed_img = replace_black_with_noise_edge_only(img, threshold, edge_width_ratio)
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, processed_img)

process_folder_with_edge_noise("c:/Malaria/Dataset/valid/cells_valid", "c:/Malaria/Dataset/valid/cells_valid_processed")
