# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.gluon as gluon

from residual import Residual
from hourglass import Hourglass

class HourglassNet(gluon.nn.HybridBlock):
    def __init__(self,nStack,nModules,nFeats,out_num):
        super(HourglassNet,self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.out_num = out_num

        self.conv1_ = gluon.nn.Conv2D(channels=64,kernel_size=(7,7),strides=(2,2),padding=(3,3))
        self.bn1 = gluon.nn.BatchNorm()
        self.r1 = Residual(in_channels=64,out_channels=128)
        self.maxpool = gluon.nn.MaxPool2D(pool_size=(2,2),strides=(2,2))
        self.r4 = Residual(in_channels=128,out_channels=128)
        self.r5 = Residual(in_channels=128,out_channels=self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []

        for i in range(self.nStack):
            _hourglass.append(Hourglass(4,self.nModules,self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(in_channels=self.nFeats,out_channels=self.nFeats))
            lin = gluon.nn.HybridSequential()
            lin.add(gluon.nn.Conv2D(channels=self.nFeats,kernel_size=1,strides=1))
            lin.add(gluon.nn.BatchNorm())
            lin.add(gluon.nn.LeakyReLU(alpha=0))
            _lin_.append(lin)
            _tmpOut.append(gluon.nn.Conv2D(channels=self.out_num,kernel_size=1,strides=1))
            if i < self.nStack - 1:
                _ll_.append(gluon.nn.Conv2D(channels=self.nFeats,kernel_size=1,strides=1))
                _tmpOut_.append(gluon.nn.Conv2D(channels=self.nFeats,kernel_size=1,strides=1))

            self.hourglass = _hourglass
            self.Residual = _Residual
            self.lin_ = _lin_
            self.tmpOut = _tmpOut
            self.ll_ = _ll_
            self.tmpOut_ = _tmpOut_

        def hybrid_forward(self,F,x):
            x = self.conv1_(x)
            x = self.bn1(x)
            x = F.relu(x)
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
                    tmpOut = self.tmpOut_[i](tmpOut)
                    x = x + ll_ + tmpOut
            
            return out