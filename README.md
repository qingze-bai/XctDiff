# XctDiff: Reconstruction of CT Images with Consistent Anatomical Structures from a Single Radiographic Projection Image

[[`Paper`](#)] [[`Dataset`](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)] [[`BibTeX`](#)]

![XctDiff Demo](assets/demo.gif?raw=true)

The **XctDiff** is capable of reconstructing CT images with consistent anatomical structures from a single radiographic projection image. This will be able to be extended to more meaningful work, such as quantitative body composition analysis, expanding medical datasets, and so on.

## Getting Started

```
conda env create -f environment.yml
```

## Training

First, we need to train a 3D perceptual compression encoder:
```
python main.py --cfg_path models/autoencoder/autoencoder_vq_32x32x32_8.yaml --gpus=2 --max_steps 80000
```
Then, we need to train an encoder to convert X-ray images to 3D features:
```
python main.py --cfg_path models/embedding/embedding_32x32x32_8.yaml --gpus=2 --max_steps 50000
```
Finally, we integrate the two components and jointly train a latent generative model
```
python main.py --cfg_path /home/first/XctDiff/models/ldm/ldm_32x32x32_8.yaml --gpus=2 --max_steps 100000
```
It is worth noting that for training, we need to specify two component pre-training weight files in the configuration file `ckpt_path`

## Inference
```
python .py --cfg_path models/ldm/ldm_32x32x32_8.yaml --input_path demo/1002.png  --ckpt_path checkpoint/xctdiff.ckpt
```


## Citation


