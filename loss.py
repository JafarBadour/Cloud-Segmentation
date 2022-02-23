import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), str(y_pred.shape) + " "+ str(y_true.shape)
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):

    
        prediction = y_pred
        prediction = prediction.cpu().detach().numpy()
        # prediction = prediction.reshape(512, 512)

        s  = prediction.max() - prediction.min()
        if s != 0:
            prediction = (prediction - prediction.min()) / s
        m = 0.5
        prediction[prediction <= m] = 0
        prediction[prediction>0]=1

        intersection = torch.logical_and(y_true, y_pred).sum()

        union = torch.logical_or(y_true, y_pred).sum()
        if union == 0:
            return torch.tensor([0]).sum()
        return 1 - intersection / union