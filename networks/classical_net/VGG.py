import torch
from torchvision import models
from torch import nn, optim
from networks.classical_net.train_base_class import Netbase


class VGGNet(Netbase):
    def __init__(self, net, class_num=6, pretrained: bool = False, **kwargs):
        super(VGGNet, self).__init__(net, class_num, pretrained, **kwargs)
        self._class_file_ = __file__
        self.net = self.getNet(net, class_num, pretrained, **kwargs)
        self.netname = net

        self.mini_batch = 128
        self.imgSize = (224, 224)

    @staticmethod
    def getNet(name, class_num, pretrained: bool, **kwargs):
        global net
        # net = torchvision.models.vgg11(pretrained=pretrained)
        # net.classifier._modules['6'] = nn.Linear(4096, int(class_num))
        if name == 'vgg11':
            net = models.vgg11(pretrained=pretrained, **kwargs)
        elif name == 'vgg11_bn':
            net = models.vgg11_bn(pretrained=pretrained, **kwargs)
        elif name == 'vgg13':
            net = models.vgg13(pretrained=pretrained, **kwargs)
        elif name == 'vgg13_bn':
            net = models.vgg13_bn(pretrained=pretrained, **kwargs)
        elif name == 'vgg16':
            net = models.vgg16(pretrained=pretrained, **kwargs)
        elif name == 'vgg16_bn':
            net = models.vgg16_bn(pretrained=pretrained, **kwargs)
        elif name == 'vgg19':
            net = models.vgg19(pretrained=pretrained, **kwargs)
        elif name == 'vgg19_bn':
            net = models.vgg19_bn(pretrained=pretrained, **kwargs)
        else: raise Exception('net name should not be None!')
        net.classifier._modules['6'] = nn.Linear(4096, int(class_num))
        return net

        pass


    def optimizer(self):
        ''' baseline: lr=0.01  mini batch = 256 '''
        # optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5 * 1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10,
        #                                                        verbose=False)  ## val_acc  , min_lr=1e-10

        optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5 * 1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=50,
                                                               verbose=False)  ## val_acc  , min_lr=1e-10

        return optimizer, scheduler

    def record_add_log(self):
        res = 'change: mini_batch = {}'.format(self.mini_batch) + ' ' +str(self.imgSize)
        res = "\n'''" + res + "'''\n"
        return res
