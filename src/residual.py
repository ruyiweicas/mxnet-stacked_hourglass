# -*- coding:utf-8 -*-
import mxnet as mx

def Residual(data, in_channels, out_channels):
    residual = data
    out = mx.sym.BatchNorm(data=data)
    out = mx.sym.relu(data=out)
    conv1 = mx.sym.Convolution(data=out,num_filter=int(out_channels/2),kernel=(1,1))
    bn1 = mx.sym.BatchNorm(data=conv1)
    bn1 = mx.sym.relu(data=bn1)
    conv2 = mx.sym.Convolution(data=bn1,num_filter=int(out_channels/2),kernel=(3,3),stride=(1,),pad=(1,))
    bn2 = mx.sym.BatchNorm(data=conv2)
    bn2 = mx.sym.relu(data=bn2)
    conv3 = mx.sym.Convolution(data=bn2,num_filter=int(out_channels),kernel=(1,1))

    if in_channels != out_channels:
        residual = mx.sym.Convolution(data=data,num_filter=int(out_channels),kernel=(1,1))

    return residual+conv3