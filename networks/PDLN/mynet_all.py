import torch
import torch.nn as nn
# from collections import OrderedDict


def _Conv2d(name_idx, inchan, out, kernel_size=3, stride=1, padding=1, bias=False):
    res = [nn.Conv2d(inchan, out, kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                     padding=(padding, padding), bias=bias),
           nn.ReLU(inplace=True)]
    return res


def _Linear(name_idx, inchan, out, bias=False):
    res = [nn.Linear(inchan, out, bias),
           nn.ReLU(inplace=True)
           ]
    return res  # nn.Sequential(*res)


class Mynet01(nn.Module):
    def __init__(self, class_num, init_weights: bool = True):
        super(Mynet01, self).__init__()
        self.ft_L = self.feature_L()
        self.ft_S = self.feature_S()
        self.trans = self._trans()
        self.classifier = self._classifier(class_num)
        if init_weights:
            self._initialize_weights()

    def feature_S(self):
        ft_L = _Conv2d('0', 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('1', 64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('2', 128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('3', 256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.AdaptiveAvgPool2d((7, 7))]

        ft_L = nn.Sequential(*ft_L)

        return ft_L

    def feature_L(self):
        # ft_M = _Conv2d('0', 3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        ft_M = _Conv2d('0', 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_M += _Conv2d('1', 64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('1', 128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('1', 128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # ft_M += _Conv2d('1', 128, 128, kernel_size=7, stride=1, padding=3, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        ft_M += _Conv2d('2', 128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('2', 256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('2', 256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # ft_M += _Conv2d('3', 256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.AdaptiveAvgPool2d((7, 7))]
        ft_M = nn.Sequential(*ft_M)
        return ft_M

    def _trans(self):
        trans = _Conv2d('0', 512 + 256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        trans += _Conv2d('1', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)

        trans += [nn.BatchNorm2d(512)]
        trans += [nn.MaxPool2d(kernel_size=2, stride=1)]

        trans += _Conv2d('0', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        trans += [nn.BatchNorm2d(512)]
        trans += [nn.MaxPool2d(kernel_size=2, stride=2)]
        trans = nn.Sequential(*trans)
        return trans

    def _classifier(self, class_num):
        classifier = _Linear('0', int(512 * 3 * 3), 4096, True)
        classifier += [nn.Dropout(0.25)]
        classifier += _Linear('0', 4096, 2048, True)
        classifier += [nn.Dropout(0.25)]

        classifier += [nn.Linear(2048, int(class_num), True), nn.Sigmoid()]  # int(class_num)
        # classifier += [nn.Sigmoid()]
        classifier = nn.Sequential(*classifier)
        return classifier

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = self.ft_L(x0)
        x2 = self.ft_S(x0)
        x = torch.cat((x, x2), dim=1)
        # print('cat:', x0.shape, x2.shape, x.shape)
        # del x0, x2
        x = self.trans(x)
        # print('trans:', x.shape)
        x = torch.flatten(x, 1)
        # print('flatten:', x.shape)
        x = self.classifier(x)
        return x


class Mynet02(nn.Module):
    def __init__(self, class_num, init_weights: bool = True):
        super(Mynet02, self).__init__()
        self.ft_L = self.feature_L()
        self.ft_S = self.feature_S()
        self.trans = self._trans()
        self.classifier = self._classifier(class_num)
        if init_weights:
            self._initialize_weights()

    def feature_S(self):
        ft_L = _Conv2d('0', 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('1', 64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('2', 128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('3', 256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.AdaptiveAvgPool2d((14, 14))]

        ft_L = nn.Sequential(*ft_L)

        return ft_L

    def feature_L(self):
        # ft_M = _Conv2d('0', 3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        ft_M = _Conv2d('0', 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_M += _Conv2d('1', 64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('1', 128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('1', 128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # ft_M += _Conv2d('1', 128, 128, kernel_size=7, stride=1, padding=3, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        ft_M += _Conv2d('2', 128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('2', 256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('2', 256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # ft_M += _Conv2d('3', 256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.AdaptiveAvgPool2d((14, 14))]
        ft_M = nn.Sequential(*ft_M)
        return ft_M

    def _trans(self):
        trans = _Conv2d('0', 512 + 256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        trans += _Conv2d('1', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)

        trans += [nn.BatchNorm2d(512)]
        trans += [nn.MaxPool2d(kernel_size=2, stride=2)]

        trans += _Conv2d('0', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        trans += [nn.BatchNorm2d(512)]
        trans += [nn.MaxPool2d(kernel_size=2, stride=2)]
        trans = nn.Sequential(*trans)
        return trans

    def _classifier(self, class_num):
        classifier = _Linear('0', int(512 * 3 * 3), 4096, True)
        classifier += [nn.Dropout(0.1)]
        classifier += _Linear('0', 4096, 2048, True)
        classifier += [nn.Dropout(0.1)]

        classifier += [nn.Linear(2048, int(class_num), True), nn.Sigmoid()]  # int(class_num)
        # classifier += [nn.Sigmoid()]
        classifier = nn.Sequential(*classifier)
        return classifier

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = self.ft_L(x0)
        x2 = self.ft_S(x0)
        x = torch.cat((x, x2), dim=1)
        # print('cat:', x0.shape, x2.shape, x.shape)
        # del x0, x2
        x = self.trans(x)
        # print('trans:', x.shape)
        x = torch.flatten(x, 1)
        # print('flatten:', x.shape)
        x = self.classifier(x)
        return x


class Mynet03(nn.Module):
    def __init__(self, class_num, init_weights: bool = True):
        super(Mynet03, self).__init__()
        self.ft_L = self.feature_L()
        self.ft_S = self.feature_S()
        self.trans = self._trans()
        self.classifier = self._classifier(class_num)
        if init_weights:
            self._initialize_weights()

    def feature_S(self):
        ft_L = _Conv2d('0', 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('1', 64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('2', 128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('3', 256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_L += _Conv2d('3', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_L += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # ft_L += [nn.AdaptiveAvgPool2d((14, 14))]

        ft_L = _Linear('0', int(512 * (224 / (2 ** 5)) ** 2), 1024, True)  #
        ft_L = nn.Sequential(*ft_L)

        return ft_L

    def feature_L(self):
        # ft_M = _Conv2d('0', 3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        ft_M = _Conv2d('0', 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('0', 64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]

        ft_M += _Conv2d('1', 64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('1', 128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('1', 128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # ft_M += _Conv2d('1', 128, 128, kernel_size=7, stride=1, padding=3, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        ft_M += _Conv2d('2', 128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('2', 256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += _Conv2d('2', 256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        ft_M += _Conv2d('3', 256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        ft_M += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # ft_M += [nn.AdaptiveAvgPool2d((14, 14))]

        ft_M = _Linear('0', int(512 * (224 / (2 ** 4)) ** 2), 1024, True)  #
        ft_M = nn.Sequential(*ft_M)
        return ft_M

    def _trans(self):
        # trans = _Conv2d('0', 512 + 256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # trans += _Conv2d('1', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        #
        # trans += [nn.BatchNorm2d(512)]
        # trans += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # trans += _Conv2d('0', 512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # trans += [nn.BatchNorm2d(512)]
        # trans += [nn.MaxPool2d(kernel_size=2, stride=2)]
        trans = _Linear('0', int(512 * (224/(2**4))**2 + 512 * (224/(2**4))**2), 4096, True)  #
        trans = nn.Sequential(*trans)
        return trans

    def _classifier(self, class_num):
        classifier = _Linear('0', int(1024*2), 4096, True)  #
        classifier += [nn.Dropout(0.1)]
        classifier += _Linear('0', 4096, 2048, True)
        classifier += [nn.Dropout(0.1)]

        classifier += [nn.Linear(2048, int(class_num), True), nn.Sigmoid()]  # int(class_num)
        # classifier += [nn.Sigmoid()]
        classifier = nn.Sequential(*classifier)
        return classifier

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = self.ft_L(x0)
        x2 = self.ft_S(x0)
        x = torch.cat((x, x2), dim=1)
        print('cat:', x0.shape, x2.shape, x.shape)
        # del x0, x2
        # x = self.trans(x)
        # print('trans:', x.shape)
        x = torch.flatten(x, 1)
        # print('flatten:', x.shape)
        x = self.classifier(x)
        return x