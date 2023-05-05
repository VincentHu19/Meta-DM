# UniSiam with Meta-DM


## Environment
* python 3.7
* pytorch 1.7
* paddlepaddle 2.3


We use 2 $\times$ A100 for training on mini-ImageNet and 4 $\times$ A100 for training on tiered-ImageNet. 

## Data Preparation
### mini-ImageNet
* download the mini-ImageNet dataset from [google drive](https://drive.google.com/file/d/1BfEBMlrf5UT4aNOoJPaa83CgbGWZAAAk/view?usp=sharing) and unzip it.

### tiered-ImageNet
* download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

### Meta-DM
To apply Meta-DM on mini-ImageNet: 


```
python generate_mini.py
```


We only use Meta-DM in meta-testing on tiered-ImageNet. To apply Meta-DM on tiered-ImageNet:


```
python generate_tiered.py
```

## Unsupervised Training

Only **DataParallel** training is supported.

Run 
```python train.py --data_path [your DATA FOLDER] --dataset [DATASET NAME] --backbone [BACKBONE] ```

For example, to train with ResNet-50 backbone on mini-ImageNet:
```
python train.py --dataset miniImageNet --backbone resnet50 --lrd_step --data_path [your mini-imagenet-folder] --save_path [your save-folder]
```


## Unsupervised Training with Distillation

Run 
```python train.py --teacher_path [your TEACHER MODEL] --data_path [your DATA FOLDER] --dataset [DATASET NAME] --backbone [BACKBONE] ```

With a teacher model, to train with ResNet-50 backbone on mini-ImageNet:
```
python train.py --dataset miniImageNet --backbone resnet50 --lrd_step --data_path [your mini-imagenet-folder] --save_path [your save-folder] --teacher_path [your teacher-model-path]
```


## Testing 
For testing, just add ```--eval_path``` to the command used for training. For example, to test with ResNet-50 backbone on mini-ImageNet: 
```
python train.py --dataset miniImageNet --backbone resnet50 --lrd_step --data_path [your mini-imagenet-folder] --save_path [your save-folder] --eval_path [your best-path]
```


For more details, please see the official implementaion of [UniSiam](https://github.com/bbbdylan/unisiam)
