# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.gluon as gluon

from src.residual import Residual
from src.hourglass import Hourglass

class HourglassNet(gluon.nn.HybridBlock):
    def __init__(self,nStack,nModules,nFeats,out_num):
        super(HourglassNet,self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.out_num = out_num

        self.conv1_ = gluon.nn.Conv2D(channels=64,kernel_size=(7,7),strides=(2,2),padding=(3,3))
        self.bn1 = gluon.nn.BatchNorm()
        self.relu = gluon.nn.LeakyReLU(alpha=0)
        self.r1 = Residual(in_channels=64,out_channels=128)
        self.maxpool = gluon.nn.MaxPool2D(pool_size=(2,2),strides=(2,2))
        self.r4 = Residual(in_channels=128,out_channels=128)
        self.r5 = Residual(in_channels=128,out_channels=self.nFeats)

        self.hourglass = gluon.nn.HybridSequential()
        self.Residual = gluon.nn.HybridSequential()
        self.lin_ = gluon.nn.HybridSequential()
        self.tmpOut = gluon.nn.HybridSequential()
        self.ll_ = gluon.nn.HybridSequential()
        self.tmpOut_ = gluon.nn.HybridSequential()

        for i in range(self.nStack):
            self.hourglass.add(Hourglass(4,self.nModules,self.nFeats))
            for j in range(self.nModules):
                self.Residual.add(Residual(in_channels=self.nFeats,out_channels=self.nFeats))
            lin = gluon.nn.HybridSequential()
            lin.add(gluon.nn.Conv2D(channels=self.nFeats,kernel_size=1,strides=1))
            lin.add(gluon.nn.BatchNorm())
            lin.add(gluon.nn.LeakyReLU(alpha=0))
            self.lin_.add(lin)
            self.tmpOut.add(gluon.nn.Conv2D(channels=self.out_num,kernel_size=1,strides=1))
            if i < self.nStack - 1:
                self.ll_.add(gluon.nn.Conv2D(channels=self.nFeats,kernel_size=1,strides=1))
                self.tmpOut_.add(gluon.nn.Conv2D(channels=self.nFeats,kernel_size=1,strides=1))
            
    def hybrid_forward(self,F,x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i*self.nModules+j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
        return out
        