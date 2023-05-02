# P>M>F with Meta-DM


## Environment
* python 3.7
* pytorch 1.7
* paddlepaddle 2.3


```
pip install -r requirements.txt
```


## Datasets
### miniImageNet
```
cd scripts
sh download_miniimagenet.sh
```

### Meta-Dataset
We follow Hu et al. by using the 'h5' files of Meta-Dataset for cross-domain testing tasks. To download them 
```
git clone https://huggingface.co/datasets/hushell/meta_dataset_h5
```


## Meta-DM
To apply Meta-DM on miniImageNet, run 
```
python generate.py
```


## Training
`--nSupport` represents the number of shots. Use `--outputs` to specify where you want to save your training results. Use `--device = cuda:i` to specify the device. 
For example, to train 5-way-5-shot with ResNet-50 as the backbone 
```
python main.py --output outputs/your_experiment_name --dataset mini_imagenet --epoch 100 --lr 5e-5 --arch dino_resnet50 --device cuda:0 --nSupport 5 --fp16
```


To use Vit-small as the backbone 
```
python main.py --output outputs/your_experiment_name --dataset mini_imagenet --epoch 100 --lr 5e-5 --arch dino_small_patch16 --device cuda:0 --nSupport 5 --fp16
```

To use Vit-base as the backbone 
```
python main.py --output outputs/your_experiment_name --dataset mini_imagenet --epoch 100 --lr 5e-5 --arch dino_base_patch16 --device cuda:0 --nSupport 5 --fp16
```

## Testing

### On miniImageNet
For example, to test your 5-wat 5-shot model with ResNet-50 as the backbone
```
python main.py --output outputs/your_experiment_name --dataset mini_imagenet --epoch 100 --lr 5e-5 --arch dino_resnet50 --device cuda:0 --nSupport 5 --fp16 --eval --resume your_best.pth
```
To test with Vit-small and Vit-base, just replace `--arch dino_resnet50` by `--arch dino_small_patch16` or `--arch dino_base_patch16` 

### Cross-domain tasks on Meta-Dataset
We recommend cross-domain testing on a single device, testing only a subset of the Meta-Dataset at the same time. Use `--test_sources your_subset` to specify the subset for testing. 
``` 
python test_meta_dataset.py --data-path /path/to/meta_dataset/ --dataset meta_dataset --arch dino_small_patch16 --deploy finetune --output outputs/your_experiment_name --resume outputs/your_experiment_name/best.pth --dist-eval --ada_steps 100 --ada_lr 0.0001 --aug_prob 0.9 --aug_types color translation --test_sources your_subset
``` 

To meta-test without fine-tuning, just replace `--deploy finetune` with `--deploy vanilla`.


For more details, please see the offical implementation of [P>M>F](https://github.com/hushell/pmf_cvpr22). 
