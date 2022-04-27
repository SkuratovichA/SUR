# SUR - __audio and visual person verification__
> Authors: Skuratovich Aliaksandr, Tikhonov Maksim
>
> Date: 25.4.2022 


## Running

Install the requirements
```bash
pip install -r requirements.txt
```

All running options listed in config hyperparams.yaml

```
python3 main.py hyperparams.yaml
```

### Training models

To repeat the experiment (and to create models based on your own data) you need to set the ``train`` param to True to classifiers you want to train models for. \
Another parameter to set is **dataset_dir** with the path to folders with training data.


### Evaluating models

In case you want to evaluate some models, set the ``eval`` param to True to corresponding classifiers whose models to be evaluated. \
Another key parameter in this case is ``eval_dir`` which contains the path to the directory with testing data.

## File structure

The implementations of three classifiers are in three corresponding folders - ``MAP`` for MAP classifier (``.wav``), ``NEURAL_PCA`` for classifiers based on NN and PCA (``.wav`` and ``.png``),
``CNN`` for the classifier based on CNN (``.png``)

## TODOS
- [x] image augmentations
- [x] audio augmentations
- [x] make a file/class/something with BGMM training
- [x] add requirements.txt
- [x] train cnn classifier
- [x] create eval pipeline. Probably add script for creation `.csv` files to train neural networks.
- [x] create train pipeline
- [ ] test everything
- [ ] add more info on how to repeat the experiment
- [ ] start the next project
