# -*- coding:utf-8 -*-
import torch
import mxnet as mx
import mxnet.gluon as gluon
import numpy as np
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE
from utils.utils import AverageMeter, Flip, ShuffleLR
from utils.eval import Accuracy, getPreds, finalPreds
# from utils.debugger import Debugger
from models.layers.FusionCriterion import FusionCriterion
import cv2
import ref
from progress.bar import Bar

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
    if split == 'train':
        model.train()
    else:
        model.eval()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []
    
    nIters = len(dataLoader)
    bar = Bar('{}'.format(opt.expID), max=nIters)
  
    for i, (input, target2D, meta) in enumerate(dataLoader):
        input_var = mx.nd.array(input)
        target_var = mx.nd.array(target2D)
        output = model(input_var)

        loss = criterion(output[0], target_var)
        for k in range(1, opt.nStack):
            loss += criterion(output[k], target_var)
        # 用于print，对Loss和Accuracy更新，print出来
        # 根据0.4.0的futurewarning更新，0.5.0以后的更新版本直接loss.data.item()取出数值
        Loss.update(loss.data[0], input.size(0))
        Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target_var.data).cpu().numpy()))
        if split == 'train':
            # train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            input_ = input.cpu().numpy()
            input_[0] = Flip(input_[0]).copy()
            inputFlip_var = torch.autograd.Variable(mx.nd.array()(input_).view(1, input_.shape[1], ref.inputRes, ref.inputRes)).float().cuda()
            outputFlip = model(inputFlip_var)
            outputFlip = ShuffleLR(Flip((outputFlip[opt.nStack - 1].data).cpu().numpy()[0])).reshape(1, ref.nJoints, ref.outputRes, ref.outputRes)
            output_ = ((output[opt.nStack - 1].data).cpu().numpy() + outputFlip) / 2
            preds.append(finalPreds(output_, meta['center'], meta['scale'], meta['rotate'])[0])
            
        Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split)
        bar.next()

    bar.finish()
    return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds
    

def train(epoch, opt, train_loader, model, criterion, optimizer):
    return step('train', epoch, opt, train_loader, model, criterion, optimizer)
    
def val(epoch, opt, val_loader, model, criterion):
    return step('val', epoch, opt, val_loader, model, criterion)
