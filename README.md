## LF-LGFA

![img](./pic/Network.jpg)

This is the PyTorch implementation of "LOCAL-GLOBAL FEATURE AGGREGATION FOR LIGHT FIELD IMAGE SUPER-RESOLUTION". 

## Contributions:

* We design a novel Local Aggregation Module to incorporate the local angular information and a Global Aggregation Module to capture global spatial information from the 4D LF data. 
* Based on these two modules, we propose a network named LF-LGFA, which achieves comparable results against state-of-the-art LFSR methods through Local-Global feature aggregation.

## Preparation:

### 1. Requirement:

We train the model using one NVIDIA RTX 2080Ti GPU card with 11GB memory .

- PyTorch 1.7.1, torchvision 0.8.2, cuda = 11.0, python = 3.7
- Matlab (For result image generation)

### 2. Compile DCN

- Cd to `code/dcn`.
- For Windows users, run `cmd make.bat`. For Linux users, run bash `bash make.sh`. The scripts will build DCN automatically and create some folders. 

### 3. Datasets

We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets.  We use Bicubic downsampling to generate low-resolution patches as the training and test datasets.  You can download our dataset via [Baidu Drive](https://pan.baidu.com/s/1okWHXUEbrAt7F3-689P_XA) (key:1234), and place the datasets to the folder `./data/`. 

## Train:

Run `train.py` to perform network training. Note that, the training settings in `train.py` should match the generated training data. Checkpoint will be saved to `./log/`.

## Test on the datasets:

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

The DCN part of our code is referred from [LF-DFnet](https://github.com/ZhengyuLiang24/LF-DFnet) . We thank the authors for sharing their codes.

## Contact

Any question regarding this work can be addressed to 3220200966@bit.edu.cn

