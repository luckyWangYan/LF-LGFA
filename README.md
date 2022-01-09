# LF-LGFA

This is the Pytorch implementation of "Local-Global Feature Aggregation For Light Field Image Super-Resolution". 


## Overview
![img](./pic/Network.jpg)


## Requirement:

- Pytorch = 1.7.1
- torchvision = 0.8.2
- cuda = 11.0
- python = 3.7
- Matlab (For result image generation)

## Compile DCN

- Cd to `code/dcn`.
- For Windows users, run `cmd make.bat`. For Linux users, run bash `bash make.sh`. The scripts will build DCN automatically and create some folders. 

## Datasets

We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets.  We use Bicubic downsampling to generate low-resolution patches as the training and test datasets.  You can download our dataset via [Baidu Drive](https://pan.baidu.com/s/1okWHXUEbrAt7F3-689P_XA) (key:1234), and place the datasets to the folder `./data/`. 

## Train

Run `train.py` to perform network training. Note that, the training settings in `train.py` should match the generated training data. Checkpoint will be saved to `./log/`.

## Test

- Run `test.py` to perform network inference. The PSNR and SSIM values of each dataset will be printed on the screen.
- Run `GenerateResultImages.m` to convert '.mat' files in `./Results/` to '.png' images to `./SRimages/`.

## Results

### **Quantitative Results**

![img](./pic/Quantitative_Results.jpg)

### **Efficiency**

![img](./pic/Efficiency.jpg)

### **Visual Comparisons**

![img](./pic/Visual_Comparisons.jpg)

## Acknowledgement

Our code is based on [LF-DFnet](https://github.com/ZhengyuLiang24/LF-DFnet). We thank the authors for sharing their codes.

## Contact

Any question regarding this work can be addressed to 3220200966@bit.edu.cn

