# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.gluon as gluon

def Residual(data, in_channels, out_channels):
    residual = data
    out = mx.sym.BatchNorm(data=data)
    out = mx.sym.relu(data=out)
    conv1 = mx.sym.Convolution(data=out,num_filter=int(out_channels/2),kernel=(1,1))
    bn1 = mx.sym.BatchNorm(data=conv1)
    bn1 = mx.sym.relu(data=bn1)
    conv2 = mx.sym.Convolution(data=bn1,num_filter=int(out_channels/2),kernel=(3,3),stride=(1,1),pad=(1,1))
    bn2 = mx.sym.BatchNorm(data=conv2)
    bn2 = mx.sym.relu(data=bn2)
    conv3 = mx.sym.Convolution(data=bn2,num_filter=int(out_channels),kernel=(1,1))

    if in_channels != out_channels:
        residual = mx.sym.Convolution(data=data,num_filter=int(out_channels),kernel=(1,1))

    return residual+conv3

def Hourglass(data,n,nModules,nFeats):
    up1 = data
    for i in range(nModules):
        for j in range(nModules):
            up1 = Residual(data=up1,in_channels=64,out_channels=128)
    # waiting for define
    pass


def HourglassNet(nStacks,nModules,nFeats,out_channels):
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data,num_filter=64,kernel=(7,7),stride=(2,2),pad=(3,3))
    bn1 = mx.sym.BatchNorm(data=conv1)
    bn1 = mx.sym.relu(data=bn1)
    r1 = Residual(data=bn1,in_channels=64,out_channels=128)
    max_pool = mx.sym.Pooling(data=r1,kernel=(2,2),stride=(2,2),pool_type='max')
    r4 = Residual(data=max_pool,in_channels=128,out_channels=128)
    r5 = Residual(data=r4,in_channels=128,out_channels=nFeats)

    _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
    for i in range(nStacks):
        _hourglass.append()