import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import torch
from config import Config


class MalariaDataset(Dataset):
    def __init__(self, datasets_dict, img_folder, is_train):
        self.datasets_dict = datasets_dict
        self.img_folder = img_folder
        self.is_train = is_train

    def __len__(self):
        return len(self.datasets_dict)

    def __getitem__(self, idx):
        dataset = list(self.datasets_dict.values())[idx]
        image_tensor = dataset.tensor
        label_map = {"uninfected": 2, "falciparum": 0, "vivax": 1}
        label = label_map[dataset.species]
        return image_tensor, label

class Dataset:
    def __init__(self, id, species, type, comment, tensor):
        self.id = id
        self.species = species
        self.type = type
        self.comment = comment
        self.tensor = tensor

    @classmethod
    def from_txt_and_image(cls, line, img_folder="dataset12/train", is_train=False):
        elements = line.strip().split(',')
        if len(elements) >= 4:
            id, species, type, comment = elements
            image_path = os.path.join(img_folder, f"{id}.jpg")
            tensor = cls.image_to_tensor(image_path, is_train=is_train)
            return cls(id, species, type, comment, tensor)
        else:
            print(f"Error: No hay suficientes elementos en la línea: {line}")
            return None

    @classmethod
    def image_to_tensor(self, image_path, is_train=False):
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # Aplicar el algoritmo de filtros
        filtered_img_np = apply_filters_to_image(img_np)
        
        # Convertir de RGB a escala de grises para la detección de bordes
        gray_filtered_image = cv2.cvtColor(filtered_img_np, cv2.COLOR_RGB2GRAY)
        
        # Aplicar detección de bordes a la imagen con filtros aplicados
        edges = cv2.Canny(gray_filtered_image, 100, 200)
        edges_expanded = np.expand_dims(edges, axis=2)
        
        # Redimensionar cada conjunto de canales por separado
        img_resized = cv2.resize(img_np, (100, 100), interpolation=cv2.INTER_AREA)
        filtered_img_resized = cv2.resize(filtered_img_np, (100, 100), interpolation=cv2.INTER_AREA)
        edges_resized = cv2.resize(edges_expanded, (100, 100), interpolation=cv2.INTER_AREA)
        
        # Concatenar todos los canales después de la redimensión
        img_stack = np.concatenate((img_resized, filtered_img_resized, np.expand_dims(edges_resized, axis=2)), axis=2)
        
        # Convertir el stack de la imagen redimensionada a un tensor PyTorch
        img_tensor = torch.from_numpy(img_stack.transpose((2, 0, 1))).float() / 255.0
        
        # Aplicar transformaciones
        if is_train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomVerticalFlip(),
            ])

        # Normalizar para ambos casos, entrenamiento y validación/test
        img_tensor = transforms.Normalize(mean=[0.5] * 7, std=[0.5] * 7)(img_tensor)
        
        return img_tensor
            
    @classmethod
    def __getitem__(self, idx):
        dataset = list(self.datasets_dict.values())[idx]
        image_path = os.path.join(self.img_folder, f"{dataset.id}.jpg")
        image_tensor = Dataset.image_to_tensor(image_path)

        # Asignar etiquetas a cada clase
        if dataset.species == "falciparum":
            label = 0
        elif dataset.species == "vivax":
            label = 1
        else:  # uninfected
            label = 2
        
        return image_tensor, label


def load_datasets_into_dict(txt_path, img_folder):
    datasets_dict = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Cargando datasets"):
            dataset = Dataset.from_txt_and_image(line, img_folder)
            if dataset:
                datasets_dict[dataset.id] = dataset
    return datasets_dict


def setup_data_loaders(train_datasets_dict, valid_datasets_dict, test_datasets_dict, train_img_folder, valid_img_folder, test_img_folder, batch_size, num_workers):
    train_dataset = MalariaDataset(train_datasets_dict, train_img_folder, is_train=True)
    val_dataset = MalariaDataset(valid_datasets_dict, valid_img_folder, is_train=False)
    test_dataset = MalariaDataset(test_datasets_dict, test_img_folder, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_loader, val_loader, test_loader

def undersample_datasets_dict(datasets_dict):
    # Contar el número de muestras por clase
    class_counts = {"falciparum": 0, "vivax": 0, "uninfected": 0}
    for dataset in datasets_dict.values():
        class_counts[dataset.species] += 1

    print("Cantidad de imágenes antes del undersampling:")
    for species, count in class_counts.items():
        print(f"{species}: {count}")

    # Determinar el número mínimo de muestras entre las clases para el undersampling
    min_samples = min(class_counts.values())

    # Nuevo diccionario de datasets después del undersampling
    undersampled_datasets_dict = {}
    new_class_counts = {"falciparum": 0, "vivax": 0, "uninfected": 0}

    for dataset_id, dataset in datasets_dict.items():
        if new_class_counts[dataset.species] < min_samples:
            undersampled_datasets_dict[dataset_id] = dataset
            new_class_counts[dataset.species] += 1

    print("\nCantidad de imágenes después del undersampling:")
    for species, count in new_class_counts.items():
        print(f"{species}: {count}")

    return undersampled_datasets_dict

def apply_filters_to_image(image):
    # Aplicar los filtros según los parámetros dados
    alpha = 1.5  # Contraste
    beta = -83   # Exposure
    saturation_scale = 2.0  # Saturation
    blur_strength = 0  # Blur
    sharpen_strength = 0  # Sharpen

    adjusted = apply_contrast(image, alpha)
    adjusted = cv2.convertScaleAbs(adjusted, alpha=1, beta=beta)
    adjusted = apply_saturation(adjusted, saturation_scale)
    adjusted = apply_blur(adjusted, blur_strength)
    adjusted = apply_sharpen(adjusted, sharpen_strength)
    
    return adjusted

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