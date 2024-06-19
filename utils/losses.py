import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifierLoss(nn.Module):

    def __init__(self, class_number=10):
        super(BinaryClassifierLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.class_number=class_number

    def forward(self, z, y, positive_only=False):
        preds = z
        y = F.one_hot(y, self.class_number) + 0.0
        # print(preds, y)
        if not positive_only:
            return torch.sum(self.criterion(preds, y), dim=1) / self.class_number
        else:
            # return torch.sum(self.criterion(preds, y), dim=1) / self.class_number
            return torch.sum(self.criterion(preds, y) * y,  dim=1)

class BinaryClassifierLoss_Negative(nn.Module):

    def __init__(self, class_number=10):
        super(BinaryClassifierLoss_Negative, self).__init__()
        self.class_number = class_number

    def forward(self, z, y):
        preds = F.sigmoid(z)
        y = F.one_hot(y, self.class_number)
        results = (1 - y) * torch.log(1 - preds)
        return - torch.sum(results, dim=1) / (self.class_number - 1)


class BinaryClassifierLoss_NoSigmoid(nn.Module):

    def __init__(self, class_number=10):
        super(BinaryClassifierLoss_NoSigmoid, self).__init__()
        self.class_number = class_number

    def forward(self, z, y):
        # preds = F.sigmoid(z)
        preds = z
        y = F.one_hot(y, self.class_number)
        results = y * torch.log(preds)
        return -torch.sum(results, dim=1)