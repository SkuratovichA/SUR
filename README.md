# SUR - __audio and visual person verification__
> Authors: Skuratovich Aliaksandr, Tikhonov Maksim
>
> Date: 25.4.2022 


## Structure
- `src/CNN/`: implementation of a Binary clasification based on a Convolutional Neural Network
- `src/MAP/`: Maximum A-posteriori Classifier based on the Bayesian Gaussian Mixture models.
- `src/NEURAL_PCA/`: Binary classification using feed-forward neural network wirh preprocessing and feature extraction with Principal Component Analysis.
- `src/models/`: Directory with ready models.
- `src/main.py`: The main script.
- `src/hyperparams.yaml`: hyperparameters to perform the training and/or evaluating.
- `dataset/`: Small dataset with `.wav` and `.png` files to train on.





## Running
> NOTE: your version of python has to be >= 3.10.0

### Install the requirements
```bash
pip install -r requirements.txt
```


### All running options listed in config hyperparams.yaml
```
python3 main.py hyperparams.yaml
```

Here are the parameters from `hyperparams.yaml`:
```yaml
default: &default
  train: False  # Train or used pretrained model
  eval: True  # evaluation is set create reports with the scores for each file from the test set. 
  model_dir: ./models  # Directory to load/store the trained model
  dataset_dir: ./dataset  # Path to a dataset to train on.
  eval_dir: ./tests  # Directory with the test set.
  GPU: 0  # If you train on the GPU machine, change this cell
  root_dir: .
  wandb_entity: <your_wandb_username_here>

CNN:
  <<: *default
  train: False
  eval: False
  model_name: CNNKyticko.pt

MAP:
  <<: *default
  dataset_dir: {non_target: ./dataset/non_target_train, target: ./dataset/target_train}
  model_name: {target: bgmm_target.pkl, non_target: bgmm_non_target.pkl}
  dev_dataset: {non_target: ./dataset/non_target_dev, target: ./dataset/target_dev}

Neural_PCA:
  <<: *default
  model_name: NeuralPCA.pt
  u_mean: ./NEURAL_PCA/u_mean.npy
```

## TODOS
- [x] image augmentations
- [x] audio augmentations
- [x] make a file/class/something with BGMM training
- [x] add requirements.txt
- [x] train cnn classifier
- [x] create eval pipeline. Probably add script for creation `.csv` files to train neural networks.
- [x] create train pipeline
- [x] test everything
- [x] add more info on how to repeat the experiment
- [ ] start the next school project :-(
