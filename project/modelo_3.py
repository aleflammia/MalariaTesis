import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.multiprocessing
torch.cuda.set_device(0)  
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')
 

class MalariaModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super(MalariaModel, self).__init__()
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  

       
        self.fc1 = nn.Linear(self._calc_fc_input(), 256)  
        self.fc2 = nn.Linear(256, num_classes)
        
        task_type = 'multiclass'

        self.train_acc = Accuracy(num_classes=num_classes, average='macro', task= task_type)
        self.train_precision = Precision(num_classes=num_classes, average='macro', task= task_type)

        self.valid_acc = Accuracy(num_classes=num_classes, average='macro', task= task_type)
        self.valid_precision = Precision(num_classes=num_classes, average='macro', task= task_type)
        self.valid_recall = Recall(num_classes=num_classes, average='macro', task= task_type)
        self.valid_f1 = F1Score(num_classes=num_classes, average='macro', task= task_type)

        self.test_acc = Accuracy(num_classes=num_classes, average='macro', task= task_type)
        self.test_precision = Precision(num_classes=num_classes, average='macro', task= task_type)
        self.test_recall = Recall(num_classes=num_classes, average='macro', task= task_type)
        self.test_f1 = F1Score(num_classes=num_classes, average='macro', task= task_type)

        self.train_predictions = []
        self.train_labels = []

        self.test_predictions = []
        self.test_labels = []

        self.val_predictions = []
        self.val_labels = []

        self.train_losses = []
        self.val_losses = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []

    def forward(self, x):
        x = self.dropout(self.pool(self.leaky_relu(self.batchnorm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.leaky_relu(self.batchnorm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.leaky_relu(self.batchnorm3(self.conv3(x)))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _calc_fc_input(self):

        with torch.no_grad():
            input_size = (1, 4, 100, 100)  # Cambia esto si el tamaño de entrada de la imagen es diferente
            x = torch.rand(*input_size)
            x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
            x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
            x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))

            return x.numel()
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_precision(preds, y)
        self.train_predictions.append(preds)
        self.train_labels.append(y)
        self.train_losses.append(loss.detach())
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.valid_precision(preds, y)
        self.valid_f1(preds, y)
        self.valid_recall(preds, y)
        self.val_predictions.append(preds)
        self.val_labels.append(y)
        self.val_losses.append(loss.detach())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_f1(preds, y)
        self.test_recall(preds, y)
        self.test_predictions.append(preds)
        self.test_labels.append(y)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()  # Computar la accuracy de entrenamiento
        train_precision = self.train_precision.compute()  # Computar la precisión de entrenamiento
        print(f"\nTrain - Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}")
        self.log("train_acc", train_acc)
        self.log("train_precision", train_precision)
        self.train_acc.reset()
        self.train_precision.reset()
        epoch_train_loss = torch.stack(self.train_losses).mean()
        self.epoch_train_losses.append(epoch_train_loss.detach().cpu().item())
        self.train_losses = [] 

    def on_validation_epoch_end(self):
        # Reset y loggear las métricas de validación al final de cada época
        valid_acc = self.valid_acc.compute()
        valid_precision = self.valid_precision.compute()
        valid_recall = self.valid_recall.compute()
        valid_f1 = self.valid_f1.compute()
        print(f"\nValidation - Accuracy: {valid_acc:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1 Score: {valid_f1:.4f}")
        self.log("valid_acc", valid_acc)
        self.log("valid_precision", valid_precision)
        self.log("valid_recall", valid_recall)
        self.log("valid_f1", valid_f1)
        self.valid_acc.reset()
        self.valid_precision.reset()
        self.valid_recall.reset()
        self.valid_f1.reset()
        epoch_val_loss = torch.stack(self.val_losses).mean()
        self.epoch_val_losses.append(epoch_val_loss.detach().cpu().item())
        self.val_losses = []


    def on_test_epoch_end(self):
        print("\n Test epoch end")

        # Print metrics at the end of each epoch
        self.log("test_acc", self.test_acc.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())
        self.log("test_f1", self.test_f1.compute())
        
        self.test_acc.reset()
        self.test_precision.reset()
        
        predictions = torch.cat(self.test_predictions, dim=0)
        labels = torch.cat(self.test_labels, dim=0)
        self.show_confusion_matrix(predictions, labels, "Test")

        self.test_predictions = []
        self.test_labels = []

    def on_fit_end(self):
        # Imprimir las matrices de confusión para el conjunto de validación y entrenamiento al final del entrenamiento
        if len(self.train_predictions) > 0 and len(self.train_labels) > 0:
            train_preds = torch.cat(self.train_predictions, dim=0)
            train_labels = torch.cat(self.train_labels, dim=0)
            self.show_confusion_matrix(train_preds, train_labels, "Train")
            self.train_predictions = []
            self.train_labels = []

        if len(self.val_predictions) > 0 and len(self.val_labels) > 0:
            val_preds = torch.cat(self.val_predictions, dim=0)
            val_labels = torch.cat(self.val_labels, dim=0)
            self.show_confusion_matrix(val_preds, val_labels, "Validation")
            self.val_predictions = []
            self.val_labels = []

        # Graficar las pérdidas de entrenamiento y validación
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_train_losses, label='Train Loss')
        plt.plot(self.epoch_val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('figures/3/Graph.png')  
        plt.close()  

        self.epoch_train_losses = []
        self.epoch_val_losses = []

    def show_confusion_matrix(self, predictions, labels, name):
        cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
        cm_norm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), normalize='true')
        cm_percentage = cm_norm * 100
        labels = (np.asarray(["{0}\n{1:.2f}%".format(count, percentage) for count, percentage in zip(cm.flatten(), cm_percentage.flatten())])).reshape(cm.shape)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=["falciparum", "vivax", "uninfected"], yticklabels=["falciparum", "vivax", "uninfected"])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'figures/3/{name}_confusion_matrix.png')
        plt.close()
