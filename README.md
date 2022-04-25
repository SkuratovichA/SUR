# SUR - __audio and visual person verification__
> Authors: Skuratovich Aliaksandr, Tikhonov Maksim
>
> Date: 25.4.2022 

## Training
Install the requirements
```bash
pip install -r requirements.txt
```

Run the training. It may take a while...
There'll be a need to have datasets you dont have, he-he.
We don't know whether we can upload the datasets on github.
```bash
python main.py mode=train
```

## Running
```bash
python main.py mode=eval
```


## TODOS
- [x] image augmentations
- [x] audio augmentations
- [] make a file/class/something with BGMM training
- [] add requirements.txt
- [] train cnn classifier
- [] create eval pipeline. Probably add script for creation `.csv` files to train neural networks.
- [] create train pipeline
- [] test everything
- [] add more info on how to repeat the experiment
- [] start the next project
