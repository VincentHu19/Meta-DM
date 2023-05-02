# Meta-GMVAE with Meta-DM


## Environment
* python 3.7
* pytorch 1.7
* tqdm

## Data
Please follow our [data preparation on Prototypical Networks with Meta-DM](https://github.com/VincentHu19/Meta-DM/tree/main/Prototypical%20Networks%20with%20Meta-DM#instructions) to download miniImageNet and apply Meta-DM. Then copy `materials` to current dir. 


## Experiments
To reproduce SimCLR features for miniImageNet 
```bash
cd simclr
python main.py --data-dir ../materials --save-dir your_simclr_experiment --feature-save-dir ../miniImageNet/data
```

To train 5-way 1/5/20/50-shot on miniImageNet
```bash
cd miniImageNet
python main.py --data-dir ./data --save-dir miniImageNet-5way-experiment
```


For more details, please see the offical implementation of [Meta-GMVAE](https://github.com/db-Lee/Meta-GMVAE). 
