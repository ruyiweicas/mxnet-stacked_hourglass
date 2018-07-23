import mxnet as mx
import mxnet.gluon as gluon
from residual import Residual
from hourglassNet import HourglassNet

net = HourglassNet(nStack=2,nModules=2,nFeats=256,out_num=16)
net.initialize()
input = mx.nd.random_normal(shape=(6,3,64,64))
output = net(input)
