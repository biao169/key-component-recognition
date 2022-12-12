'''PDLN: Parallel deep learning networks'''

import torch
from torch import optim

from networks.classical_net.train_base_class import Netbase
from networks.PDLN import mynet_all as Mynet
import config_train as config

class PDLN(Netbase):
    def __init__(self, net, class_num=6, pretrained=False, **kwargs):
        super(PDLN, self).__init__(net, class_num, **kwargs)
        self._class_file_ = __file__
        self.net = self.getNet(net, class_num, pretrained, **kwargs)
        self.netname = net
        self.imgSize = (224,224)
        # print(self.net)
        # exit()
        self.mini_batch = 128
        self.trainbatch_size = 16

    @staticmethod
    def getNet(name, class_num, pretrained: bool, **kwargs):
        global net
        if name == 'mynet01':
            net = Mynet.Mynet01(class_num, **kwargs)
        elif name == 'mynet02':
            net = Mynet.Mynet02(class_num, **kwargs)
        elif name == 'mynet03':
            net = Mynet.Mynet03(class_num, **kwargs)


        else:
            raise Exception('net name should not be None!')
        return net
        pass

    def set_param(self, param:config):
        param.mini_batch = self.mini_batch
        param.trainbatch_size = self.trainbatch_size
        return param

    def optimizer(self):
        optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5 * 1e-3)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6,
                                                               verbose=False)  ## val_acc  , min_lr=1e-10
        return optimizer, scheduler

    def record_add_log(self):
        res = "\n\nnetPath = r'" + __file__ + "'\n"
        res += "''' PDLN: mynet '''\n"
        with open(r'./networks/PDLN/mynet_all.py', mode='r', encoding='utf-8') as f3:
            # print('__file__:', self._class_file_)
            start_p = False
            lines = f3.readlines()
            for line in lines:
                if 'class M{net}(nn.Module):'.format(net=self.netname[1:]) in line:
                    start_p = True
                    res += line
                    continue
                if start_p and '(nn.Module)' in line:
                    print('class M{net}(nn.Module):'.format(net=self.netname[1:]))
                    start_p = False
                    break
                if start_p:
                    res += line
        return res
        pass
