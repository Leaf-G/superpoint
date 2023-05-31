import timm
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np
from models.BTL import BTL




class SuperPointOptimized(nn.Module):
    def __init__(self):
        super(SuperPointOptimized, self).__init__()

        # Backbone
        backbone = timm.create_model('mobilenetv2_140', pretrained=True, in_chans=1)
        layers = list(backbone.children())

        layers_new = []
        layers_new.extend(layers[0:2])
        layers_new.extend(list(layers[2]))

        self.encoder_truncated = torch.nn.Sequential(*layers_new[0:5])

        # Variables
        c4 = 48
        c5 = 256
        d1 = 256
        det_h = 65

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)


        # ReLU
        self.relu = torch.nn.ReLU(inplace=True)
        # self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.BTL = BTL(20, 256)

    def forward(self, x):
        # Extract features
        fts = self.encoder_truncated(x)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(fts)))
        semi = self.bnPb(self.convPb(cPa))

        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(fts)))
        desc = self.bnDb(self.convDb(cDa))

        # Normalize descriptors
        # dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        # desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        desc = desc.squeeze()
        desc = self.BTL(desc)

        output = {'semi': semi, 'desc': desc}
        self.output = output

        return output
if __name__ == '__main__':

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SuperPointOptimized()
  model = model.to(device)

  # check keras-like model summary using torchsummary
  from torchsummary import summary
  summary(model, input_size=(1, 224, 224))