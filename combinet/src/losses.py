import torch
import torch.nn as nn
from combinet.src.dataset.camvid import CLASS_WEIGHT_CAMVID
from combinet.src.dataset.bacteria import CLASS_WEIGHT_BACTERIA

class SegmentationLoss(nn.Module):
    def __init__(self, args):
      super(SegmentationLoss, self).__init__()
      self.args = args
      if self.args.dataset == "camvid":
        self.class_weight = CLASS_WEIGHT_CAMVID
        self.loss = nn.NLLLoss(weight = self.class_weight)
        self.iou = IoULoss()
      elif self.args.dataset == "bacteria":
        self.class_weight = CLASS_WEIGHT_BACTERIA
        self.loss = nn.NLLLoss(weight = self.class_weight)
        self.iou = IoULoss()
        self.dice = DiceLoss()

    def forward(self, outs, targets):
        if self.args.dataset == "camvid":
          self.loss.weight = self.loss.weight.to(outs.device)
          loss = self.loss(torch.log(outs+1e-8), targets) + self.iou(outs, targets, ignore_last_index = True)
        elif self.args.dataset == "bacteria":
          self.loss.weight = self.loss.weight.to(outs.device)
          loss = self.loss(torch.log(outs+1e-8), targets) + self.iou(outs, targets) +  self.dice(outs, targets)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target, ignore_last_index = False):
        if predict.shape != target.shape:
          if target.is_cuda:
            target_one_hot = torch.cuda.LongTensor(predict.shape).zero_()
          else:
            target_one_hot = torch.LongTensor(predict.shape).zero_()
          target = target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        if ignore_last_index:
            predict = predict[:,:-1,:,:]
            target = target[:,:-1,:,:]
        intersection = torch.sum(predict * target, dim = [0, 2, 3])
        union = torch.sum(predict, dim = [0, 2, 3]) +  torch.sum(target, dim = [0, 2, 3])
        loss = -torch.log(torch.mean((2.* intersection + self.smooth)/ (union+self.smooth), dim = 0)+1e-8)
        return loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1.):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target, ignore_last_index = False):
        if predict.shape != target.shape:
          if target.is_cuda:
            target_one_hot = torch.cuda.LongTensor(predict.shape).zero_()
          else:
            target_one_hot = torch.LongTensor(predict.shape).zero_()
          target = target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        if ignore_last_index:
            predict = predict[:,:-1,:,:]
            target = target[:,:-1,:,:]
        intersection = torch.sum(predict * target, dim = [0, 2, 3])
        union = torch.sum(predict, dim = [0, 2, 3]) +  torch.sum(target, dim = [0, 2, 3]) - intersection
        # The factor 2. is a mistake that got copy pasted from the dice loss
        # it should have been -torch.log(torch.mean((1.* intersection + self.smooth)/ (union+self.smooth), dim = 0)+1e-8), oh well...
        loss = -torch.log(torch.mean((2.* intersection + self.smooth)/ (union+self.smooth), dim = 0)+1e-8)
        return loss