# mxnet version stacked hourglass network

## Network
This repo is a mxnet-gluon version of stacked hourglass network.Stacked hourglass network is widelt used
in human pose estimation.

##Requirements
mxnet-cu80==1.3\
progress\
opencv-python\
h5py


## Datasets
Mpii human pose dataset.

## Parameters Config
Train parameter can be set in opts.py and ref.py. You can set parameters in these two 
files. The details of how to set parameters for stacked hourglass network is here [https://github.com/umich-vl/pose-hg-train/blob/master/src/opts.lua].
You can set all parameters as you prefer in opts and ref.

##Train
When you train stacked hourglass Network or some network based on this repo, just 
>python main.py 

to train your model.You can use some flags to change my default.Such as 
>python main.py -nEpochs 120

to set epochs.You can know more details in opts.py. 