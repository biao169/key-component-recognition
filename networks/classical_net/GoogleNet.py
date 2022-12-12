import torch
from torchvision import models
from torch import nn, optim
from networks.classical_net.train_base_class import Netbase


class GoogleNet(Netbase):
    def __init__(self, net, class_num=6, pretrained: bool = False, **kwargs):
        super(GoogleNet, self).__init__(net, class_num, **kwargs)
        self._class_file_ = __file__
        self.net = self.getNet(net, class_num, pretrained, **kwargs)
        self.netname = net
        self.imgSize = (224, 224)

    @staticmethod
    def getNet(name, class_num, pretrained: bool, **kwargs):
        global net
        # net = torchvision.models.vgg11(pretrained=pretrained)
        # net.classifier._modules['6'] = nn.Linear(4096, int(class_num))
        if name == 'googlenet':
            net = models.googlenet(pretrained=pretrained, **kwargs)

        else: raise Exception('net name should not be None!')
        net.classifier._modules['6'] = nn.Linear(4096, int(class_num))
        return net
        pass

    def optimizer(self):
        optimizer = optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5 * 1e-4)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6,
                                                               verbose=False)  ## val_acc  , min_lr=1e-10
        return optimizer, scheduler
