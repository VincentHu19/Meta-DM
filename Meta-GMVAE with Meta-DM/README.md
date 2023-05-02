# Meta-GMVAE: Mixture of Gaussian VAE for Unsupervised Meta-Learning


## Environment
* python 3.7
* pytorch 1.7
* tqdm

## Data
Please follow our [instruction on Prototypical Networks with Meta-DM] to download miniImageNet and apply Meta-DM. 
```
cd ../
```

## Experiment
To reproduce **Mini-ImageNet 5-way experiment** for Meta-GMVAE, run the following code:
```bash
cd mimgnet
python main.py --data-dir DATA DIRECTORY (e.g. /home/dongbok/data/mimgnet/) --save-dir SAVE DIRECTORY (e.g. /home/dongbok/mimgnet-5way-experiment)
```

(Optional) To reproduce SimCLR features for Mini-ImageNet, run the following code:
```bash
cd simclr
python main.py --data-dir DATA DIRECTORY (e.g. /home/dongbok/data/imgnet/) --save-dir SAVE DIRECTORY (e.g. /home/dongbok/simclr-experiment) --feature-save-dir FEATURE SAVE DIRECTORY (e.g. /home/dongbok/data/mimgnet)
```


For more details, please see the offical implementation of [Meta-GMVAE](https://github.com/db-Lee/Meta-GMVAE). 
