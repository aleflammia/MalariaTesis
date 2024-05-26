import torch
import os
import re
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint 
from data_loader import setup_data_loaders, load_datasets_into_dict, undersample_datasets_dict
from config import Config 
from sklearn.model_selection import KFold

import importlib

model_name = 'modelo_4' 
model_module = importlib.import_module(model_name)

MalariaModel = getattr(model_module, 'MalariaModel')

torch.cuda.set_device(0) 
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    dirpath=f'checkpoints/{model_name}',
    filename='malaria-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # Guarda el mejor modelo según el monitor
    mode='min', 
)
def stratified_split(full_dataset, labels):
    keys = list(full_dataset.keys())  # Obtiene las claves actuales del diccionario
    label_counts = {label: labels.count(label) for label in set(labels)}
    indices_by_class = {label: [keys[i] for i, lbl in enumerate(labels) if lbl == label] for label in set(labels)}

    folds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Realizamos la división estratificada para cada clase
    for _ in range(5):
        train_idx, test_valid_idx = [], []
        for label, indices in indices_by_class.items():
            np.random.shuffle(indices)
            train_count = int(0.8 * len(indices))
            valid_test_count = len(indices) - train_count
            valid_count = test_count = valid_test_count // 2

            train_idx.extend(indices[:train_count])
            test_valid_idx.extend(indices[train_count:])

        # Dividir test_valid_idx más en test y valid
        np.random.shuffle(test_valid_idx)
        mid_point = len(test_valid_idx) // 2
        valid_idx = test_valid_idx[:mid_point]
        test_idx = test_valid_idx[mid_point:]

        folds.append((train_idx, valid_idx, test_idx))
    
    return folds
    
def save_indices_to_txt(indices, file_path):
    with open(file_path, 'w') as f:
        for index in indices:
            f.write(f"{index}\n")

def train_model(model, train_loader, val_loader, test_loader, max_epochs, fold):  
    model.current_fold = fold
    print(f"Comenzando el entrenamiento para el pliegue {fold + 1}...")
    trainer = Trainer(max_epochs=max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback])

    # Entrenar el modelo
    trainer.fit(model, train_loader, val_loader)
    print(f"Entrenamiento para el pliegue {fold + 1} completado.")

    # Probar el modelo y recoger resultados
    print(f"Comenzando la evaluación en el conjunto de prueba para el pliegue {fold + 1}...")
    test_result = trainer.test(model, dataloaders=test_loader)
    print(f"Evaluación para el pliegue {fold + 1} completada.\n")

    # Intentar acceder a las métricas de manera segura
    metrics = {
        'test_acc': float(test_result[0].get('test_acc', 0)),
        'test_precision': float(test_result[0].get('test_precision', 0)),
        'test_recall': float(test_result[0].get('test_recall', 0)),
        'test_f1': float(test_result[0].get('test_f1', 0)),
        'test_specificity': float(test_result[0].get('test_f1', 0)),
    }

    # Guarde las métricas
    fold_dir = f'Tesis/cross_validation/results/{model_name}/fold_{fold+1}'
    
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    with open(f'{fold_dir}/metrics_fold_{fold+1}.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Métricas guardadas para el pliegue {fold + 1}.\n")

if __name__ == "__main__":

    print(torch.__version__)
    print(torch.version.cuda)
    config = Config()

    # Cargar todo el conjunto de datos
    full_dataset = load_datasets_into_dict('Conjunto/conjunto.txt', 'Conjunto')
    print("Claves disponibles en full_dataset:", list(full_dataset.keys())[:10])  # Imprime algunas claves para verificación

    labels = [dataset.species for dataset in full_dataset.values()]  # Asegúrate de que esta línea extraiga correctamente las etiquetas

    # Preparar y obtener las divisiones estratificadas
    folds = stratified_split(full_dataset, labels)

    # Iterar sobre cada fold
    for fold, (train_idx, valid_idx, test_idx) in enumerate(folds):
        print(f"Realizando la iteración {fold + 1} de la validación cruzada")

        print(f"Fold {fold + 1}: Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

        print("Índices de entrenamiento generados:", train_idx[:10]) 

        save_indices_to_txt(train_idx, f'Tesis/cross_validation/results/{model_name}/fold_{fold+1}/train_indices.txt')
        save_indices_to_txt(valid_idx, f'Tesis/cross_validation/results/{model_name}/fold_{fold+1}/valid_indices.txt')
        save_indices_to_txt(test_idx, f'Tesis/cross_validation/results/{model_name}/fold_{fold+1}/test_indices.txt')

        # Seleccionar subconjuntos de datos según los índices
        train_datasets_dict = {id: full_dataset[id] for id in train_idx}
        valid_datasets_dict = {id: full_dataset[id] for id in valid_idx}
        test_datasets_dict = {id: full_dataset[id] for id in test_idx}

        train_datasets_dict = undersample_datasets_dict(train_datasets_dict)
        valid_datasets_dict = undersample_datasets_dict(valid_datasets_dict)
        test_datasets_dict = undersample_datasets_dict(test_datasets_dict)

        # Configurar los DataLoaders
        train_loader, val_loader, test_loader = setup_data_loaders(
            train_datasets_dict, valid_datasets_dict, test_datasets_dict,
            'Conjunto', 'Conjunto', 'Conjunto', config.batch_size, config.num_workers
        )

        # Inicializar modelo y entrenar
        model = MalariaModel(num_classes=3)
        train_model(model, train_loader, val_loader, test_loader, config.max_epochs, fold)