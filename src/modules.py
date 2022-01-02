from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder


class BirdSpeciesDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def train_dataloader(self):
        train_dataset = ImageFolder(self.config.paths.data.train, self.augmentation)
        return DataLoader(train_dataset, **self.config.params.train_loader)

    def val_dataloader(self):
        val_dataset = ImageFolder(self.config.paths.data.val, self.transform)
        return DataLoader(val_dataset, **self.config.params.val_loader)

    def test_dataloader(self):
        test_dataset = ImageFolder(self.config.paths.data.test, self.transform)
        return DataLoader(test_dataset, **self.config.params.test_loader)


class BirdSpeciesModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build_model()
        self.criterion = eval(self.config.params.loss)()
        # self.save_hyperparameter(config)

    def build_model(self):
        self.backbone = timm.create_model(
            self.config.params.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 325),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        out = self.head(features)
        out = F.log_softmax(out, dim=1)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = eval(self.config.params.optimizer.name)(
            self.parameters(), **self.config.params.optimizer.params
        )
        scheduler = eval(self.config.params.scheduler.name)(
            optimizer, **self.config.params.scheduler.params
        )
        return [optimizer], [scheduler]
