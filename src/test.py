import mxnet as mx
import mxnet.gluon as gluon
from residual import Residual
from hourglassNet import HourglassNet
from hourglass import Hourglass

input = mx.nd.random_normal(shape=(6,3,64,64))
"""
net = gluon.nn.HybridSequential()
net.add(gluon.nn.Conv2D(channels=64,kernel_size=(7,7),strides=(2,2),padding=(3,3)))
net.add(gluon.nn.BatchNorm())
net.add(gluon.nn.LeakyReLU(alpha=0))
net.add(Residual(64,128))
net.add(gluon.nn.MaxPool2D(pool_size=(2,2),strides=(2,2)))
net.add(Residual(128,128))
net.add(Residual(128,256))
net.add(Hourglass(n=4,nModules=2,nFeats=256))
for i in range(2):
    net.add(Residual(256,256))

net.initialize()
output = net(input)
print(output.shape)
# net = Hourglass(n=4,nModules=2,nFeats=256)
"""
net = HourglassNet(nStack=2,nModules=2,nFeats=256,out_num=16)
net.initialize()
output = net(input)
print(output[0].shape)
