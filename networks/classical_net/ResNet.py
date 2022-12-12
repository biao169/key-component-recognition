import torch
from torchvision import models
from torch import optim
from networks.classical_net.train_base_class import Netbase

import config_train as config




class ResNet(Netbase):
    def __init__(self, net, class_num=6, pretrained: bool = False, **kwargs):
        super(ResNet, self).__init__(net, class_num, pretrained, **kwargs)
        self._class_file_ = __file__
        self.net = self.getNet(net, class_num, pretrained, **kwargs)
        self.netname = net

        self.mini_batch = 128
        self.trainbatch_size = 2**3
        self.imgSize = (224, 224)
        # print( self.net)
        # exit()

    def set_param(self, param: config):
        param.mini_batch = self.mini_batch
        param.trainbatch_size = self.trainbatch_size
        return param

    @staticmethod
    def getNet(name, class_num, pretrained: bool, **kwargs):
        global net
        # net = torchvision.models.vgg11(pretrained=pretrained)
        # net.classifier._modules['6'] = nn.Linear(4096, int(class_num))
        if name == 'resnet18':
            net = models.resnet18(pretrained=pretrained, **kwargs)
        elif name == 'resnet34':
            net = models.resnet34(pretrained=pretrained, **kwargs)
        elif name == 'resnet50':
            net = models.resnet50(pretrained=pretrained, **kwargs)
        elif name == 'resnet101':
            net = models.resnet101(pretrained=pretrained, **kwargs)
        elif name == 'resnet152':
            net = models.resnet152(pretrained=pretrained, **kwargs)
        else:
            raise Exception('net name should not be None!')
        num_ftrs = net.fc.in_features
        bias_ftrs = net.fc.bias
        if bias_ftrs is not None:
            bias_ftrs = True
        else:
            bias_ftrs = False
        net.fc = torch.nn.Linear(num_ftrs, int(class_num), bias=bias_ftrs)
        return net
        pass

    def optimizer(self):
        ''' baseline: lr=0.1  mini batch = 128 '''
        # optimizer = optim.SGD(self.net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1 * 1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=50,
        #                                                        verbose=False)  ## val_acc  , min_lr=1e-10

        optimizer = optim.SGD(self.net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1 * 1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=50,
                                                               verbose=False)  ## val_acc  , min_lr=1e-10

        ''' mini batch = 128 '''
        return optimizer, scheduler

    def record_add_log(self):
        res = 'mini_batch = {}'.format(self.mini_batch)
        res = "\n'''" + res + "'''\n"
        return res
