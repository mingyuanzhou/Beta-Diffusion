# Code for Beta Diffusion published in NeurIPS 2023

```bibtex
@inproceedings{zhou2023beta,
    title={Beta Diffusion},
    author={Mingyuan Zhou and  Tianqi Chen and Zhendong Wang and Huangjie Zheng},
    booktitle = {Neural Information Processing Systems},
    year={2023}
}
```

# Experiments with Beta Diffusion

This folder contains the code for Beta Diffusion. The 'toy_experiment' folder contains Jupyter notebooks to reproduce the results on two synthetic datasets using CPU. The 'image_experiment' folder contains PyTorch code to reproduce the results of Beta Diffusion on CIFAR10 unconditional image generations using GPUs. The instructions to run Beta Diffusion for image generations are provided as follows:

## Installation

First install the packages needed to run the code. Note that you may choose a PyTorch version that is compatible with your local environment. 

```sh
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install click pillow scipy psutil requests tqdm imageio pyspng
```
After setting up the environment, go to one of the folder. For example:
```sh
cd BetaDiffusion/image_experiment
```

## Data Preparation
You may leverage the data already downloaded in the local folder. Alternatively, you may prepare the data following the steps provided below:


### Copy the preprocessed data
Download the CIFAR10 data:
```sh
wget -P downloads/cifar10/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
```
which will put the 'cifar10-32x32.zip' file into the 'datasets' folder 


## Training
To train Beta Diffsuion model using 4 GPUs on CIFAR10:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4  train.py --outdir=betadiff-train-runs/ --data=datasets/cifar10-32x32.zip --cond=False --arch=ddpmpp --batch=512 --precond=betadiff --lr=2e-4 --Shift=0.60 --Scale=0.39 --sigmoid_start=10 --sigmoid_end=-13 --sigmoid_power=1 --lossType='KLUB' --eta=10000
```
where you can adjust lossType as 'KLUB' or 'KLUB-AS' 

## Sampling 
Sample 100 example images using 200 NFEs
```
python -m torch.distributed.run --standalone --nproc_per_node=4 generate.py --steps=200 --outdir=plots/generated_images --network=plots/checkpoint_beta_KLUB/network-snapshot-200000.pkl --seeds=0-99 --batch=100
```


## FID Evaluation

For FID evaluation, you can use either '--ref=refs/cifar10-32x32.npz' or '--ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'. For the first option, you will need to run
```sh
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```
which will put the 'cifar10-32x32.npz' file into the 'fid-refs' folder.



To evaluate the trained Beta Diffsuion model, first place both the checkpoint 'filename.pkl', which stores the model, and the json file 'training_options.json', which stores model paramerers used for training, into a folder, such as 'plots/checkpoint_beta_KLUB/', and then run the following code:

```bash
#Generate 50000 images, using NFEs=200
python -m torch.distributed.run --standalone --nproc_per_node=4 generate.py --steps=200 --outdir=plots/images --network=plots/checkpoint_beta_KLUB/network-snapshot-200000.pkl --seeds=0-49999
#Calculate the FID of x0_hat
python -m torch.distributed.run --standalone --nproc_per_node=4 fid.py calc --images=plots/images --ref=$fid_file
#Calculate the FID of z_0
python -m torch.distributed.run --standalone --nproc_per_node=4 fid.py calc --images=plots/images_1 --ref=$fid_file
```
