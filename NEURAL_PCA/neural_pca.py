# File: cnn.py
# Author: Skuratovich Aliaksandr <xskura01@vutbr.cz>
# Date: 27.4.2022, 3.46 AM

from safe_gpu import safe_gpu
gpu_owner = safe_gpu.GPUOwner(1)

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(levelname)s): %(funcName)s:%(lineno)d %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

DEBUG = False
EPOCHS = 1 if DEBUG else 1000
logger.disabled = not DEBUG


class ConvertedDataset(Dataset):
    def __init__(self, d, labels):
        self.data = d
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class ImageTransform:
    def __init__(self, is_train: bool):
        if is_train:
            self.transform = transforms.Compose([
                iaa.Sequential([
                    iaa.Resize((80, 80)),
                    iaa.Sometimes(0.8, iaa.GaussianBlur(sigma=(0, 2.0))),
                    iaa.Sometimes(0.8, iaa.PerspectiveTransform(scale=(0.0, 0.03))),
                    iaa.Fliplr(0.8),
                    iaa.Sometimes(0.8, iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
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
                    iaa.Resize((80, 80)),
                    iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.0, 0.02))),
                    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 3.0))),
                    iaa.Sometimes(0.1, iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True)),
                    iaa.Sometimes(0.1,
                                  iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.6))
                                  ),
                    iaa.Sometimes(0.05, iaa.SomeOf((0, 2), [
                        iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)),  # sharpen images
                        iaa.Emboss(alpha=(0, 0.7), strength=(0, 0.1)),  # emboss images
                    ]))
                ]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


def convert_pil_to_tensor_and_transform(img_path: str, trans=None, size=(80, 80)):
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
    image = trans(image).squeeze(0).reshape(size[0] * size[1]).numpy()
    return image


class PCADataset(pl.LightningDataModule):
    def __init__(self, root_dir: str = '.', batch_size: int = 4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = 1
        self.classes = 1

        def get_images(csv_name, is_train=False):
            a = pd.read_csv(csv_name)
            datas = []
            labels = []
            for index in range(len(a)):
                img_path = os.path.join(root_dir, a.iloc[index, 0])
                label = np.array(int(a.iloc[index, 1]))
                image = convert_pil_to_tensor_and_transform(img_path, trans=ImageTransform(is_train=is_train))
                datas.append(image)
                labels.append(label)
            return torch.Tensor(np.r_[datas]), torch.Tensor(np.r_[labels]).to(torch.int32)

        def pca_transform(datas, u, mean_):
            d_u = torch.tensor((datas.numpy() - mean_.numpy()).dot(u.T.numpy()))
            return d_u

        train_d, train_l = get_images(os.path.join(root_dir, 'all.csv'), is_train=True)
        val_d, val_l = get_images(os.path.join(root_dir, 'dev_PCA.csv'), is_train=False)
        self.shape = len(train_d)

        v, s, u = torch.linalg.svd(train_d, full_matrices=False)
        mean = train_d.mean()

        self.U = u
        self.mean = mean
        self.train_dataset = ConvertedDataset(pca_transform(train_d, self.U, self.mean), train_l)
        self.val_dataset = ConvertedDataset(pca_transform(val_d, self.U, self.mean), val_l)

    def get_input_shape(self):
        return self.shape

    def get_u_mean(self):
        return self.U, self.mean

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return dataloader


class NeuralPCA(pl.LightningModule):
    def __init__(self):
        #self.save_hyperparameters()
        super().__init__()
        self.activ = nn.Sigmoid()
        self.activ2 = nn.GELU()  # I LOVE GELU. (4.22 AM)

        IN_SHAPE = 239
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=IN_SHAPE),  # normalize input features

            nn.Linear(in_features=IN_SHAPE, out_features=32),
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

    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        out = self(x).squeeze(1)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00009, weight_decay=0.01, amsgrad=True)  # the best?
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


def get_basic_callbacks(checkpoint_interval) -> list:
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = pl.callbacks.ModelCheckpoint(filename='epoch{epoch:03d}',
                                                 auto_insert_metric_name=False,
                                                 save_top_k=1,
                                                 every_n_epochs=checkpoint_interval)
    return [ckpt_callback, lr_callback]


def main(hparams):
    # need to know U, mean from the train dataset

    dataset_personality =  "u_mean.npy" # i mean the largest eigenvalues, mean

    if hparams["train"]:
        data = PCADataset(root_dir=hparams["dataset_dir"], batch_size=32)
        u, mean = data.get_u_mean()
        # store U, mean
        with open(dataset_personality, "wb") as f:
            np.save(f, u)
            np.save(f, mean)

        with open(dataset_personality, "wb") as f:
            np.save(file=f, arr=u)
            np.save(file=f, arr=u)

        name = hparams["model_name"]
        # set train logger
        train_logger = WandbLogger(offline=True,
                                   name=name,
                                   project="NeuralPCA",
                                   save_dir=os.path.join(hparams["root_dir"], "wandb"),
                                   entity=hparams["wandb_entity"])
        os.environ["WANDB_MODE"] = "offline"

        model = NeuralPCA()
        train_logger.watch(model)

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            callbacks=get_basic_callbacks(checkpoint_interval=10),
            default_root_dir=hparams["root_dir"],
            gpus=hparams["GPU"],
            num_sanity_val_steps=1,
            log_every_n_steps=10,
            logger=train_logger
        )

        trainer.fit(model, data)
        save_path = os.path.join(hparams["model_dir"], name)
        torch.save(model.state_dict(), save_path)

    #if DEBUG:
    if True:
        with open(dataset_personality, "rb") as f:
            u = np.load(f)
            mean = np.load(f)

        path = os.path.join(hparams["model_dir"], hparams["model_name"])
        logger.debug(f"model path: {path}")
        model = NeuralPCA()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()

        img_path = "../dataset/target_dev/m421_03_p01_i0_0.png"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        image = (convert_pil_to_tensor_and_transform(img_path, transform))  # numpy form
        image = torch.tensor((image - mean).dot(u.T)).unsqueeze(0)
        print(f"for {img_path}")
        print(f"Score: {model(image).detach().ravel()[0]}")


if __name__ == "__main__":

    # only CPU tests
    hparams = {"train": False,
               "model_dir": "..",
               "model_name": "neural_pca.pt",
               "root_dir": "../",
               "wandb_entity": "skuratovich",
               "dataset_dir": "../dataset/",
               "GPU": 1
               }
    main(hparams)

