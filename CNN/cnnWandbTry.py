import os
from typing import Optional, Union
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import Accuracy
from PIL import Image
import PIL
import imgaug as ia
from imgaug import augmenters as iaa
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from pprint import pprint
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import Callback


class SURDataset(Dataset):
    def __init__(self, root_dir, csv, transform=None):
        self.annotations = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
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
                    iaa.Resize((80, 80)),
                    iaa.Sometimes(0.8, iaa.GaussianBlur(sigma=(0, 2.0))),
                    iaa.Fliplr(0.9),
                    iaa.Sometimes(0.9, iaa.Affine(rotate=(-65, 65), mode='edge')),
                    iaa.Sometimes(0.8, iaa.PerspectiveTransform(scale=(0.0, 0.10))),
                    iaa.Sometimes(0.9, iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True)),
                    iaa.Sometimes(0.8, iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.6))),
                    iaa.Sometimes(0.1, iaa.SomeOf((0, 2), [
                        iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)),  # sharpen images
                        iaa.Emboss(alpha=(0, 0.7), strength=(0, .5)),  # emboss images
                    ]))]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
            ])
        else:
            self.transform = transforms.Compose([
                iaa.Sequential([
                    iaa.Resize((80, 80)),
                    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 3.0))),
                    iaa.Fliplr(0.2),
                    iaa.Sometimes(0.3, iaa.Affine(rotate=(-65, 65), mode='edge')),
                    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.0, 0.10))),
                    iaa.Sometimes(0.3, iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True)),
                    iaa.Sometimes(0.3, iaa.blend.Alpha((0.0, 1.0), first=iaa.Add(20), second=iaa.Multiply(0.4))),
                    iaa.Sometimes(0.05, iaa.SomeOf((0, 2), [
                        iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5)),  # sharpen images
                        iaa.Emboss(alpha=(0, 0.7), strength=(0, .5)),  # emboss images
                    ]))
                ]).augment_image,
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class SmallDataset(pl.LightningDataModule):
    def __init__(self, root_dir: str = '.', batch_size: int = 32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = 1
        self.train_dataset = SURDataset(root_dir=root_dir, csv='train1.csv', transform=ImageTransform(is_train=True))
        self.val_dataset = SURDataset(root_dir=root_dir, csv='dev1.csv', transform=ImageTransform(is_train=False))
        self.classes = 1

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return dataloader


class VeriOverFit(pl.LightningModule):
    def __init__(self):
        # self.save_hyperparameters()
        super().__init__()
        ##
        #        self.linear = nn.ReLU()
        #        self.activ = nn.GELU()
        #
        #        self.model = nn.Sequential(
        #
        #            nn.Dropout(0.01),  # self-augmentation
        #
        #            nn.Conv2d(1, 8, kernel_size=5, padding=3), self.activ,
        #            nn.MaxPool2d(2, 2),
        #            nn.BatchNorm2d(num_features=8),
        #            nn.Dropout(0.1),
        #
        #            nn.Conv2d(8, 16, kernel_size=3, padding=1), self.activ,
        #            nn.MaxPool2d(2, 2),
        #            nn.BatchNorm2d(num_features=16),
        #            nn.Dropout(0.3),
        #
        #            nn.Conv2d(16, 32, kernel_size=3, padding=1), self.activ,
        #            nn.MaxPool2d(2, 2),
        #            nn.BatchNorm2d(num_features=32),
        #            nn.Dropout(0.4),
        #
        #            nn.Conv2d(32, 64, kernel_size=3, padding=1), self.activ,
        #            nn.MaxPool2d(2, 2),
        #            nn.BatchNorm2d(num_features=64),
        #            nn.Dropout(0.4),
        #
        #            nn.Flatten(),
        #            nn.BatchNorm1d(num_features=1600),
        #            nn.Linear(1600, 128), nn.GELU(),
        #            nn.Linear(128, 1), nn.Sigmoid()
        #        )
        #
        #        self.loss = torch.nn.BCELoss()
        #        self.accuracy = Accuracy()
        #
        ##

        self.linear = nn.ReLU()
        self.activ = nn.GELU()

        self.model = nn.Sequential(

            nn.Dropout(0.05),  # self-augmentation

            nn.Conv2d(1, 8, kernel_size=5, padding=3), self.activ,
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout(0.09),

            nn.Conv2d(8, 16, kernel_size=3, padding=1), self.activ,
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout(0.35),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), self.activ,
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.45),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), self.activ,
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.45),

            nn.Flatten(),
            nn.BatchNorm1d(num_features=1600),
            nn.Linear(1600, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid()
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
        # self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

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

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00000008, weight_decay=0.00156,
                                        momentum=0.999)  # the best?

        # optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00001, weight_decay=0.0001, momentum=0.99) # the best?
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True) # the best?
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=1, weight_decay=0.0) # the best?
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.000001, weight_decay=0.001, nesterov=True, momentum=0.9) # the best?
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.8,
            patience=1,
            threshold=0.0001,
            min_lr=0.e-14,  # almost zero lol
            verbose=True
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val loss',
            'interval': 'epoch',
            'frequency': 2,
        }
        return {"optimizer": optimizer, "lr_scheduler_HLR": lr_scheduler_config}


class LogPredictionsCallback(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        if np.random.randint(low=0, high=10) != 7:
            return

        if batch_idx == np.random.randint(low=0, high=batch_idx + 1):
            n = 5
            x, y = batch
            images = [img for img in x[:n]]

            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]

            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            self.logger.log_table(key='sample_table', columns=columns, data=data)


def get_basic_callbacks(checkpoint_interval, logger) -> list:
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = pl.callbacks.ModelCheckpoint(filename='epoch{epoch:03d}',
                                                 auto_insert_metric_name=False,
                                                 save_top_k=1,
                                                 every_n_epochs=checkpoint_interval)
    prediction_callback = LogPredictionsCallback()

    return [ckpt_callback, lr_callback, prediction_callback]


def main(hparams):
    if hparams["train"]:
        name = hparams["model_name"]
        # set logger
        logger = WandbLogger(offline=True,
                             name=name,
                             project="cnnMerlin",
                             save_dir=os.path.join(hparams["root_dir/"], 'wandb'),
                             entity=hparams["wandb_entity"])
        os.environ["WANDB_MODE"] = "offline"

        data = SmallDataset(root_dir=hparams["root_dir"], batch_size=4)
        model = VeriOverFit()
        # logger.watch(model)
        trainer = pl.Trainer(
            max_epochs=50000,
            callbacks=get_basic_callbacks(logger),
            default_root_dir=hparams["root_dir"],
            gpus=hparams["GPU"],
            num_sanity_val_steps=1,
            log_every_n_steps=6,
            logger=logger
        )
        print("3")
        trainer.fit(model, data)
        print(f"Saving model: {name}")
        torch.save(model.state_dict(), os.path.join(hparams["model_dir"], "/" + name))


if __name__ == "__main__":
    # TODO: move to main
    from safe_gpu import safe_gpu  # noqa

    gpu_owner = safe_gpu.GPUOwner(1)

    hparams = {"train": True, }
    main(hparams)
