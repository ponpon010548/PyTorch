import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class VGG(nn.Module, neck, classes, shape):  
    #neck : VGG13, 16... 
    #classes : number of class
    #shape : image shape

    def __init__(self):
        super(self).__init__()
        self.neck = neck
        self.classifier = nn.Sequential(OrderDict([
                            ("l1", nn.Linear(512 * shape[0] * shape[1] / 1024, 4096))
                            ("c_act1", nn.ReLU(inplace = True))
                            ("l2", nn.Linear(4096, 4096))
                            ("c_act2", nn.ReLU(inplace = True))
                            ("l3", nn.Linear(4096, classes)))
                            ])
    def forward(self, x):
        x = self.neck(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#BN = batch normalize
def convLayer(cfg, in_channel, BN = True):
    layer = []
    in_channel = in_channel
    c, p, bn, act = 1, 1, 1, 1          #layer NO.
    for dim in cfg:
        if dim == 'P':                  #MaxPool layer
            name = "pool" + str(p)
            p += 1
            layer.append((name, nn.MaxPool2d(kernel_size = 2, stride = 2)))
            continue

        conv = nn.Conv2d(in_channel, dim, kernel_size = 3, padding = 1)    #conv layer
        name = "conv" + str(c)
        c += 1
        name2 = "act" + str(act)
        in_channel = dim             #in_channel for next conv layer
        if not BN:
            layer.append((name, conv))
            layer.append((name2, nn.ReLU(inplace = True)))   #save memory
            continue
        layer.append((name, conv))
        name = "BN" + str(bn)
        bn += 1
        layer.append((name, nn.BatchNorm2d(dim)))
        layer.append((name2, nn.ReLU(inplace = True)))
    return nn.Sequential(OrderdDict(layer))              #model with layers' name

