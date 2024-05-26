import torch
import os
import re
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint 
from data_loader import setup_data_loaders, load_datasets_into_dict, undersample_datasets_dict
from config import Config 

import importlib

model_name = 'modelo_4' 
model_module = importlib.import_module(model_name)

MalariaModel = getattr(model_module, 'MalariaModel')
model = MalariaModel(num_classes=3)

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

def train_model(model, train_loader, val_loader, test_loader, max_epochs):  
    trainer = Trainer(max_epochs=max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path

    match = re.search(r"malaria-(\d+)-([\d.]+)\.ckpt", os.path.basename(best_model_path))
    if match:
        epoch, val_loss = match.groups()
        print(f"La mejor configuración se consiguió en la época {epoch} con una pérdida de validación de {val_loss}")

    best_model = MalariaModel.load_from_checkpoint(checkpoint_callback.best_model_path)

    # Ahora usa esta nueva instancia para el testeo
    trainer.test(best_model, dataloaders=test_loader)

if __name__ == "__main__":

    print(f"Entrenando el modelo: {model_name}")

    config = Config()  # Se crea una instancia de Config para usar sus atributos

    # Carga tu diccionario de conjuntos de datos de entrenamiento, validación y prueba
    train_datasets_dict = load_datasets_into_dict(config.train_txt_path, config.train_data_path)
    valid_datasets_dict = load_datasets_into_dict(config.valid_txt_path, config.valid_data_path)
    test_datasets_dict = load_datasets_into_dict(config.test_txt_path, config.test_data_path)

    # Aplica undersampling a tus datasets si es necesario
    train_datasets_dict = undersample_datasets_dict(train_datasets_dict)
    valid_datasets_dict = undersample_datasets_dict(valid_datasets_dict)
    test_datasets_dict = undersample_datasets_dict(test_datasets_dict)


    # Configura tus DataLoaders
    train_loader, val_loader, test_loader = setup_data_loaders(
        train_datasets_dict, valid_datasets_dict, test_datasets_dict,
        config.train_data_path, config.valid_data_path, config.test_data_path, config.batch_size, config.num_workers
    )

    # Entrena tu modelo
    train_model(model, train_loader, val_loader, test_loader, config.max_epochs)
