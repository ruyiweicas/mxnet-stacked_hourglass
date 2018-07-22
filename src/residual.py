# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.gluon as gluon

class Residual(gluon.nn.HybridBlock):
    def __init__(self,in_channels,out_channels):
        super(Residual,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = gluon.nn.BatchNorm()
        
        self.conv1 = gluon.nn.Conv2D(channels=int(out_channels),kernel_size=(1,1))
        self.bn2 = gluon.nn.BatchNorm()
        self.conv2 = gluon.nn.Conv2D(channels=int(out_channels),kernel_size=(3,3),strides=(1,1),padding=(1,1))
        self.bn3 = gluon.nn.BatchNorm()
        self.conv3 = gluon.nn.Conv2D(channels=int(out_channels),kernel_size=(1,1))
        
        if in_channels!=out_channels:
            self.conv4 = gluon.nn.Conv2D(channels=int(out_channels),kernel_size=(1,1))
        
    def hybrid_forward(self,F,x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)

        if self.in_channels != self.out_channels:
            residual = self.conv4(x)
        return out + residual