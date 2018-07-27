import os
import time
import datetime
import mxnet as mx
import mxnet.gluon as gluon

import src.opts as opts
import src.ref as ref
from .hourglassNet import HourglassNet
from .utils.utils import adjust_learning_rate
from .data.mpii import MPII
from .utils.utils import AverageMeter, Flip, ShuffleLR
from .utils.logger import Logger
from .utils.eval import Accuracy, getPreds, finalPreds
from progress.bar import Bar
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

    mseloss = gluon.loss.L2Loss()

    val_dataset = MPII(opt,'val')
    train_dataset = MPII(opt,'train')

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
    loss = 0
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []
    nIters = len(train_loader)
    bar = Bar('{}'.format(opt.expID), max=nIters)

    for epoch in range(1, opt.nEpochs + 1):
        for i, (input, target, meta) in enumerate(train_loader):
            if epoch % opt.snapshot == 0:
                model.hybridize()
            output = model(input)
            for k in range(1, opt.nStack):
                loss += mseloss(output[k], target)
            Loss.update(loss.asnumpy(), input.shape[0])
            Acc.update(Accuracy((output[opt.nStack - 1]).asnumpy(), (target).asnumpy()))
            loss.backward()
            trainer.step(batch_size=opt.trainBatch)

            Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(
                epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split="train")
            bar.next()
        bar.finish()

        log_dict_train = {'Loss': Loss.avg, 'Acc': Acc.avg}
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if epoch % opt.snapshot == 0:
            model.export('hourglass2D',epoch=epoch)
