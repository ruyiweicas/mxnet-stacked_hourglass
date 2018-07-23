# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.gluon as gluon
import os
import time
from residual import Residual

class Hourglass(gluon.nn.HybridBlock):
    def __init__(self,n,nModules,nFeats):
        super(Hourglass,self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        self.up1_ = gluon.nn.HybridSequential()
        self.low1_ = gluon.nn.HybridSequential()
        self.low2_ = gluon.nn.HybridSequential()
        self.low3_ = gluon.nn.HybridSequential()
        
        for i in range(self.nModules):
            self.up1_.add(Residual(in_channels=self.nFeats,out_channels=self.nFeats))

        self.low1 = gluon.nn.MaxPool2D(pool_size=(2,2),strides=(2,2))
        for i in range(self.nModules):
            self.low1_.add(Residual(in_channels=self.nFeats,out_channels=self.nFeats))
        
        if self.n > 1:
            self.low2 = Hourglass(n=n-1,nModules=self.nModules,nFeats = self.nFeats)
        else:
            for i in range(self.nModules):
                self.low2_.add(Residual(in_channels=nFeats,out_channels=nFeats))

        for i in range(self.nModules):
            self.low3_.add(Residual(in_channels=self.nFeats,out_channels=nFeats))
        
        #用反卷积代替upsampling，可以按情况试试
        #self.up2 = gluon.nn.Conv2DTranspose(channels=self.nFeats,kernel_size=(2,2),strides=(1,1))
        
    def hybrid_forward(self,F,x):
        up1 = x
        for i in range(self.nModules):
            up1 = self.up1_[i](up1)
        
        low1 = self.low1(x)
        for i in range(self.nModules):
            low1 = self.low1_[i](low1)
        
        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for i in range(self.nModules):
                low2 = self.low2_[i](low2)
        
        low3 = low2
        for i in range(self.nModules):
            low3 = self.low3_[i](low3)
        # gluon的nn里没有upsample，混合直接调用ndarray的sample
        up2 = F.UpSampling(low3,sample_type='bilinear',scale=2)

        return up1 + up2 


    
