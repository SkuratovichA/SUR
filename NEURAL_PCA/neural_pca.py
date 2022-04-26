
from wandb import wandb


zaebalsya = PCA_dataset('.')
IN_FEATURES = zaebalsya.get_input_shape()
print(IN_FEATURES)

eval = False
name = "PCA_NN_dev_and_train.pt"
EPOCHS = 350


import torch
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from PIL import Image
import wandb
from imgaug import augmenters as iaa
import os
import pandas as pd

import pytorch_lightning as pl

# TODO: move to main
gpu_owner = safe_gpu.GPUOwner(1)
import torch.nn.functional as F

class ImageTransform:
    def __init__(self, is_train: bool):
        if is_train:
            self.transform = transforms.Compose([
                iaa.Sequential([
                iaa.Resize((80,80)),
                iaa.Sometimes(0.8, iaa.GaussianBlur(sigma=(0,2.0))),
                iaa.Sometimes(0.8, iaa.PerspectiveTransform(scale=(0.0, 0.06))),
                iaa.Fliplr(0.8),
                iaa.Sometimes(0.5, iaa.Affine(rotate=(-3, 3), mode='edge')),
                iaa.Sometimes(0.8, iaa.AddToHueAndSaturation(value=(-10,10), per_channel=True)),
                iaa.Sometimes(0.2, 
                    iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.6))
                ),
                ]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
        ])
        else:
            self.transform = transforms.Compose([
                iaa.Sequential([
                    iaa.Resize((80,80)),
                    iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.0, 0.03))),
                    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0,3.0))),
                    iaa.Sometimes(0.1, iaa.AddToHueAndSaturation(value=(-20,20), per_channel=True)),
                    iaa.Sometimes(0.1, 
                        iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.6))
                    ),
                    iaa.Sometimes(0.05, iaa.SomeOf((0, 2), [
                        iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 0.7), strength=(0, 0.1)), # emboss images
                    ]))
                ]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
            ])
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class ConvertedDataset(Dataset):
    def __init__(self, d, labels):
        self.data = d
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.labels[index])

    
def convert_PIL_to_tensor_and_transform(img_path: str, trans=None, size=(80,80)):
    # image = io.imread(img_path)
    image = Image.open(img_path)
    if trans is None:
        trans = transforms.Compose([
                iaa.Sequential([iaa.Resize(size)]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
        ])
    image = np.array(image)
    image = trans(image).squeeze(0).reshape(size[0]*size[1]).numpy()
    return image


class PCA_dataset(pl.LightningDataModule):
    def __init__(self, root_dir: str='.', batch_size: int=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers=1
        self.classes = 1

        def get_images(csvname, is_train=False):
            a = pd.read_csv(csvname)
            datas = []
            labels = []
            for index in range(len(a)):
                img_path = os.path.join(root_dir, a.iloc[index, 0])
                label = np.array(int(a.iloc[index, 1]))
                image = convert_PIL_to_tensor_and_transform(img_path, trans=ImageTransform(is_train=is_train))
                datas.append(image)
                labels.append(label)
            return torch.Tensor(np.r_[datas]), torch.Tensor(np.r_[labels]).to(torch.int32)

        def pca_transform(datas, U, mean):
            d_U = torch.tensor((datas.numpy() - mean.numpy()).dot(U.T.numpy()))
            return d_U
        
        train_d, train_l = get_images('train.csv', is_train=True)
        val_d, val_l = get_images('dev.csv', is_train=False)
        self.shape = len(train_d)
        
        V, S, U = torch.linalg.svd(train_d, full_matrices=False)
        mean = train_d.mean()

        self.U = U
        self.mean = mean
        self.train_dataset = ConvertedDataset(pca_transform(train_d, U, mean), train_l)
        self.val_dataset = ConvertedDataset(pca_transform(val_d, U, mean), val_l)
    
    def get_input_shape(self):
        return self.shape

    def get_U_mean(self):
        return self.U, self.mean
    
    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return dataloader


class NeuralPCA(pl.LightningModule):
    def __init__(self):
        self.save_hyperparameters()
        super().__init__()
        self.activ = nn.Sigmoid()
        self.activ2 = nn.GELU()
        
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=IN_FEATURES), # normalize input features

            nn.Linear(in_features=IN_FEATURES, out_features=32),
            self.activ,
            
            nn.Linear(in_features=32, out_features=1),
            self.activ,
        )

        self.loss = torch.nn.BCELoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('train accuracy', acc)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('val accuracy', acc)
        self.log('val loss', loss)
        return preds

    def transform_for_test(self, dato):
        transformed = torch.tensor((dato.numpy() - self.mean.numpy()).dot(self.U.T.numpy()))

    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        out = self(x).squeeze(1)
        #preds, _ = out.max(1)
        loss = self.loss(out, y.to(torch.float32))
        preds = (out > .5).to(torch.int32)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00009, weight_decay=0.0001, amsgrad=True) # the best?
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.8,
            patience=1,
            threshold=1e-15)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {"optimizer": optimizer, "lr_scheduler_HLR": lr_scheduler_config}


if not eval:
    wandb_logger = WandbLogger(log_model='all', name=name, project="Neural_PCA")


def get_basic_callbacks(checkpoint_interval: int=1) -> list:
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = pl.callbacks.ModelCheckpoint(filename='epoch{epoch:03d}',
                                    auto_insert_metric_name=False,
                                    save_top_k=1,
                                    every_n_epochs=checkpoint_interval)
    
    return [ckpt_callback, lr_callback]


def main():

    if not eval:
        data = PCA_dataset(root_dir='.', batch_size=16)
        U, mean = data.get_U_mean()
        model = NeuralPCA()
        wandb_logger.watch(model)
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            callbacks=get_basic_callbacks(),
            default_root_dir='.',
            gpus=0,
            strategy=None,
            num_sanity_val_steps=0,
            log_every_n_steps=10,
            logger=wandb_logger
        )
        trainer.fit(model, data)
        print(f"Saving model: {name}")
        torch.save(model.state_dict(), name)
        print("saved")
        wandb.finish()
        print("finishing wandb")
    else:
        print("Loading model")
        model = NeuralPCA()
        model.load_state_dict(torch.load(name))
        model.eval()
        print("model is ready!?")
main()
