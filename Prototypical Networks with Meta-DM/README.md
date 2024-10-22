# Prototypical Networks with Meta-DM

### Results

1-shot: 59.3% (49.4% without Meta-DM)

5-shot: 72.3% (68.2% without Meta-DM)

## Environment

* python 3.7
* pytorch 1.7
* paddlepaddle 2.3

## Data Preparation

1. Download the images: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

2. Make a folder `materials/images` and put those images into it.

`--gpu` to specify device for program.

### To apply Meta-DM

```
python generate.py
```


## Experiments
### 1-shot Train

```
python train.py
```

### 1-shot Test

```
python test.py
``` 

### 5-shot Train

```
python train.py --shot 5 --train-way 20 --save-path ./save/proto-5
```

### 5-shot Test

```
python test.py --load ./save/proto-5/max-acc.pth --shot 5
```
