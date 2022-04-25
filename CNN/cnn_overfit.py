# -*- coding: utf-8 -*-
"""cnn_overfit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yHTdUbqT0af5aiUY5aDPd_csHIfIoxYz
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install -q pytorch_lightning wandb torchmetrics 
# from google.colab import drive
# drive.mount('/content/drive')
# %cp -r "drive/MyDrive/SUR/SUR_projekt2021-2022"/* .
# !pip install git+https://github.com/aleju/imgaug --yes

import os
from typing import Optional, Union

import torchvision
import PIL
import tqdm
import wandb
import torch
import numpy as np
import imgaug as ia
import pandas as pd
from torch import nn
from PIL import Image
from pprint import pprint
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

# from safe_gpu import safe_gpu
# gpu_owner = safe_gpu.GPUOwner(0)

import torch.nn.functional as F


class SURDataset(Dataset):
    def __init__(self, root_dir, csv, transform=None):
        self.annotations = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        #image = io.imread(img_path)
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = np.array(image)
            image = self.transform(image)
        return (image, y_label)


class ImageTransform:
    def __init__(self, is_train: bool):
        if is_train:
            self.transform = transforms.Compose([
                iaa.Sequential([
                    iaa.Resize((80,80)),
                    iaa.Sometimes(0.8, iaa.GaussianBlur(sigma=(0,2.0))),
                    iaa.Fliplr(0.9),
                    iaa.Sometimes(0.8, iaa.Affine(rotate=(-65,65), mode='edge')),
                    iaa.Sometimes(0.8, iaa.PerspectiveTransform(scale=(0.0, 0.10))),
                    iaa.Sometimes(0.8, iaa.AddToHueAndSaturation(value=(-20,20), per_channel=True)),
                    iaa.Sometimes(0.8, iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.6))),
                    iaa.Sometimes(0.1, iaa.SomeOf((0, 2), [
                        iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 0.7), strength=(0, .5)), # emboss images
                ]))]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                # transforms.Normalize(mean=[133.8628, 104.0229, 106.3728],
                                    #  std=[54.8065, 56.6426, 54.9152]),
            ])
        else:
            self.transform = transforms.Compose([
                iaa.Sequential([
                    iaa.Resize((80,80)),
                    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0,3.0))),
                    iaa.Fliplr(0.2),
                    iaa.Sometimes(0.3, iaa.Affine(rotate=(-65,65), mode='edge')),
                    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.0, 0.10))),
                    iaa.Sometimes(0.3, iaa.AddToHueAndSaturation(value=(-20,20), per_channel=True)),
                    iaa.Sometimes(0.3, iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.4))),
                    iaa.Sometimes(0.05, iaa.SomeOf((0, 2), [
                        iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 0.7), strength=(0, .5)), # emboss images
                    ]))
                ]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                # transforms.Normalize(mean=[133.8628, 104.0229, 106.3728],
                                    #  std=[54.8065, 56.6426, 54.9152]),
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class SmallDataset(pl.LightningDataModule):
    def __init__(self, root_dir: str='.', batch_size: int=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers=1
        self.train_dataset = SURDataset(root_dir=root_dir, csv='train_cnn.csv', transform=ImageTransform(is_train=True))
        self.val_dataset = SURDataset(root_dir=root_dir, csv='dev_cnn.csv', transform=ImageTransform(is_train=False))
        self.classes = 1
    
    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return dataloader









pl.utilities.seed.seed_everything(42)
import wandb
wandb.login()

class VeriOverFit(pl.LightningModule):
    def __init__(self):
        self.save_hyperparameters()
        super().__init__()
        self.activ = nn.CELU()
        self.non_lin_activ = nn.GELU()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features=32),
            self.activ,
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=32),
            self.activ,
            nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding =1),
            nn.BatchNorm2d(num_features=16),
            self.activ,
            nn.MaxPool2d(2, 2), # h: 40
        
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features=16),
            self.activ,
            nn.Conv2d(16, 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=8),
            self.activ,
            nn.Conv2d(8, 4, kernel_size = 3, stride = 1, padding =1),
            nn.BatchNorm2d(num_features=4),
            self.activ,
            nn.MaxPool2d(2, 2), # h: 40
            
            nn.Flatten(),
            nn.Linear(1600, 512), # ??? 80*80*1 ?
            self.non_lin_activ,
            nn.Linear(512, 64), # ??? 80*80*1 ?
            self.non_lin_activ,
            nn.Linear(64, 1),
            nn.Sigmoid(),
            # nn.Dropout2d(p=dropout) 0.1
        )

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log('train accuracy', acc)
        self.log('train loss', loss)
        # self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log('val accuracy', acc)
        self.log('val loss', loss)
        #self.log_dict({'val/loss': loss, 'val/acc': acc})

        return preds

    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        out = self(x).squeeze(1)

        #preds, _ = out.max(1)
        loss = self.loss(out, y.to(torch.float32))
        preds = (out > .5).to(torch.int32)
        
        # print(preds.shape, y.shape)
        acc = self.accuracy(preds, y)

        return preds, loss, acc
    
    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.GELU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.3) # the best?
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.999,
            patience=5,
            threshold=1e-15)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val loss',
            'interval': 'epoch',
            'frequency': 2,
        }
        return {"optimizer": optimizer, "lr_scheduler_HLR": lr_scheduler_config}



wandb_logger = WandbLogger(log_model='all', name='RMSprop2', project="pytorch_lightning_test", offline=True)

from pytorch_lightning.callbacks import Callback
 
class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        if batch_idx == np.random.randint(low=0, high=batch_idx+1):
            n = 20
            x, y = batch
            images = [img for img in x[:n]]

            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]

            # Option 1: log images with `WandbLogger.log_image`
            # wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)

# log_predictions_callback = LogPredictionsCallback()

def get_basic_callbacks(checkpoint_interval: int=1) -> list:
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = pl.callbacks.ModelCheckpoint(filename='epoch{epoch:03d}',
                                    auto_insert_metric_name=False,
                                    save_top_k=1,
                                    every_n_epochs=checkpoint_interval)
    prediction_callback = LogPredictionsCallback()
    
    return [ckpt_callback, lr_callback, prediction_callback]


def test_saving(model):
    torch.save(model.state_dict(), "OverFitDumbass.pt")
    test_model = VeriOverFit()
    test_model.load_state_dict(
        torch.load("OverFitDumbass.pt")
    )
    test_model.eval()

def main():

    MODELNAME = 'SmalloverfittedStupidModel1'
    eval = False
    name = "OverFitDumbass_RAdam.pt"

    if not eval:
        data = SmallDataset(root_dir='.', batch_size=32)
        model = VeriOverFit()
        wandb_logger.watch(model)
        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=get_basic_callbacks(),
            default_root_dir='.',
            gpus=0,
            strategy=None,
            num_sanity_val_steps=1,
            log_every_n_steps=10,
            logger=wandb_logger
        )
        trainer.fit(model, data)
        print(f"Saving model: {name}")
        torch.save(model.state_dict(), name)
        wandb.finish()
    else:
        print("Loading model")
        model = VeriOverFit()
        model.load_state_dict(torch.load(name))
        model.eval()
        ## todo: evaluate a model
        # eval_model(lambda model, input: model(input), model)

main()

# ## saving (test)
# torch.save(model.state_dict(), "OverFitDumbass.pt")
# test_model = VeriOverFit()
# test_model.load_state_dict(
#     torch.load("OverFitDumbass.pt")
# )
# test_model.eval()
