# mxnet version stacked hourglass network

## Network
This repo is a mxnet-gluon version of stacked hourglass network.Stacked hourglass network is widelt used
in human pose estimation.

## Requirements
The code has been tested with CUDA 8.0 and ubuntu 16.04.

python2.7\
mxnet-cu80==1.3\
progress\
opencv-python\
h5py

If you use python3,there has some bugs in decode path of images.The info is in mpii.loadimage.When using python3,you can
write a function to convert the path loaded from .h5 from illegal to legal.

## Datasets
Mpii human pose dataset.

## Parameters Config
Train parameter can be set in opts.py and ref.py. You can set parameters in these two 
files. The details of how to set parameters for stacked hourglass network is here [https://github.com/umich-vl/pose-hg-train/blob/master/src/opts.lua].
You can set all parameters as you prefer in opts and ref.

## Train
When you train stacked hourglass Network or some network based on this repo, just 
>python main.py 

to train your model.You can use some flags to change my default.Such as 
>python main.py -nEpochs 120

to set epochs.You can know more details in opts.py. 

## Thanks
Thanks Alejandro Newell's torch version of stacked hourglass network code.
