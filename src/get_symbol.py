# -*- coding:utf-8 -*-

# deprecated 
# 直接转成mxnet的symbol由于hourglass结构问题写起来不如gluon方便
import mxnet as mx
import mxnet.gluon as gluon

def Residual(data,in_channels,out_channels):
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

    return residual + conv3

def Hourglass(data,n,nModules,nFeats):
    up1 = data
    for i in range(nModules):
        up1 = Residual(data=up1,in_channels=nFeats,out_channels=nFeats)
    # maxpooling下采样
    low1 = mx.sym.Pooling(data=up1,kernel=(2,2),stride=(2,2),pool_type='max')
    for i in range(nModules):
        low1 = Residual(data=low1,in_channels=nFeats,out_channels=nFeats)
    if n > 1:
        low2 = Hourglass(data=low1,n=n-1,nModules=nModules,nFeats=nFeats)
    else:
        low2 = low1
        for i in range(nModules):
            low2 = Residual(data=low2,in_channels=nFeats,out_channels=nFeats)
    low3 = low2
    for i in range(nModules):
        low3 = Residual(data=low3,in_channels=64,out_channels=64)
    # 再把模块尺度上采样出来
    up2 = mx.sym.UpSampling(data=low3,scale=2)

    return up1 + up2

def HourglassNet(nStacks,nModules,nFeats,out_channels):
    data = mx.sym.Variable('data')
    conv1 = mx.sym.Convolution(data=data,num_filter=64,kernel=(7,7),stride=(2,2),pad=(3,3))
    bn1 = mx.sym.BatchNorm(data=conv1)
    bn1 = mx.sym.relu(data=bn1)
    r1 = Residual(data=bn1,in_channels=64,out_channels=128)
    max_pool = mx.sym.Pooling(data=r1,kernel=(2,2),stride=(2,2),pool_type='max')
    r4 = Residual(data=max_pool,in_channels=128,out_channels=128)
    r5 = Residual(data=r4,in_channels=128,out_channels=nFeats)
    x = r5
    out = []
    
    for i in range(nStacks):
        hg = Hourglass(data=x,n=4,nModules=nModules,nFeats=nFeats)
        ll = hg
        for j in range(nModules):
            ll = Residual(data=ll,in_channels=nFeats,out_channels=nFeats)
        pass