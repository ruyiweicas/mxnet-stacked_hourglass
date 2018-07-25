import os
import time
import datetime
import mxnet as mx
import mxnet.gluon as gluon

import opts
import ref
from hourglassNet import HourglassNet
from utils.utils import adjust_learning_rate
from datasets.fusion import Fusion
from datasets.h36m import H36M
from datasets.mpii import MPII
from utils.logger import Logger
from train import train, val

from model import getModel, saveModel

from utils.logger import Logger


import scipy.io as sio

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    opt = opts.opts().parse()
    now = datetime.datetime.now()
    logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

    if opt.loadModel != 'none':
        sym, arg_params, aux_params = mx.model.load_checkpoint("hourglass", 60)
        model = mx.mod.Module(symbol=sym, context=mx.gpu(), data_names=["input_var"], label_names=["target_var"])
        model.bind(for_training=True, data_shapes=[("input_var", (opt.trainBatch,3,ref.inputRes,ref.inputRes))])
        model.set_params(arg_params,aux_params)
    else:
        model = HourglassNet(nModules=2,nFeats=256,nStack=2,out_num=256)
        model.collect_params().initialize(mx.init.Xavier())

    # heatmap的计算是直接求均方误差
    mseloss = gluon.loss.L2Loss()

    val_dataset = gluon.data.Dataset(MPII(opt,'val',returnMeta=True))
    train_dataset = gluon.data.Dataset(MPII(opt,'train',returnMeta=True))

    val_loader = gluon.data.DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = int(ref.nThreads)
    )

    train_loader = gluon.data.DataLoader(
        train_dataset, 
        batch_size = opt.trainBatch, 
        shuffle = True if opt.DEBUG == 0 else False,
        num_workers = int(ref.nThreads)
    )
    trainer = gluon.Trainer(model.collect_params(),'rmsprop',{'learning_rate': 2e-5,
                                                                'gamma1': 0.99,
                                                                'gamma2': 0.0,
    
                                                                'epslion': 1e-8})
    # train，调用train方法
    for epoch in range(1, opt.nEpochs + 1):
        for i, (input, target, meta) in enumerate(train_loader):
            