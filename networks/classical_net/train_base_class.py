import datetime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn, optim

from utils_tool.log_printFormat import train_val_alldata as saveTraindata
from Control import controlManage2 as controlManage
from config_train import record2Text


class Netbase:
    def __init__(self, net, class_num=6, pretrained: bool = False, **kwargs):
        self.isremove_all = True
        self.test_loader = None
        self.train_loader = None
        self.trainbatch_size = 0
        self.testbatch_size = 0
        self.net = self.getNet(net, class_num, pretrained, **kwargs)
        self.netname = net
        self._class_file_ = __file__
        self.log_name = None
        self.netPath = None

        if pretrained:
            self.log_ispre = 'Pre_'
            print('采用 预训练 赋值模式！')
        else:
            self.log_ispre = 'noPre_'
            print('采用 非预训练 赋值模式！')
        self.exit = False


    @staticmethod
    def getNet(name, class_num, pretrained: bool, **kwargs):
        global net
        net = models.vgg11(pretrained=pretrained)
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
        net.classifier._modules['6'] = nn.Linear(4096, int(class_num))
        return net

        pass

    def run(self, device, trainDataset, testDataset=None, param=None):
        torch.manual_seed(param.random_seed)
        torch.cuda.manual_seed(param.random_seed)
        np.random.seed(param.random_seed)  # 用于numpy的随机数

        self.trainConfig(param.trainOutput_root, trainDataset, param.trainbatch_size,
                         testDataset, param.torch_shuffle, param.drop_last, param.tips)

        optimizer, scheduler = self.optimizer()
        start_epoch, stop_epoch = 0, param.epoch
        if param.train_continue:
            optimizer, epoch = self.continue_training_load_state(optimizer, param)
            start_epoch, stop_epoch = epoch, param.epoch + epoch
        while True:
            try:
                # torch.cuda.empty_cache()
                if param.mini_batch == 0:
                    end_epoch = self.train(self.train_loader, device, optimizer, start_epoch, stop_epoch,
                                           param.save_frequency, param.startSave_acc
                                           , scheduler=scheduler, online_test=param.nline_test)
                else:
                    end_epoch = self.train_minibatch(self.train_loader, device, optimizer, start_epoch, stop_epoch,
                                                     param.save_frequency, param.startSave_acc
                                                     , scheduler=scheduler, mini_batch=param.mini_batch,
                                                     drop_last=param.drop_last, online_test=param.online_test)
                if testDataset is not None:
                    torch.cuda.empty_cache()
                    torch.cuda.init()
                    torch.cuda.empty_cache()
                    begin = datetime.datetime.now()
                    loss, acc = self.test_module(net=self.net, device=device, test_loader=self.test_loader)
                    end = datetime.datetime.now()
                    print(self.netname, '测试集 测试结果：', 'test_loss={} , test_acc={}'.format(loss, acc))
                    if not self.exit:
                        train_data = {'index': end_epoch,
                                      'test_loss': loss, 'test_acc': acc,
                                      'spend_time': float((end - begin).total_seconds())}
                        train_data_savefile = os.path.join(self.netPath, self.log_name.format('data') + '.csv')
                        saveTraindata(train_data_savefile, train_data)
                break
            except RuntimeError as e:
                print(e)
                if 'CUDA out of memory' in str(e):
                    self.trainbatch_size = int(self.trainbatch_size / 2)
                    if self.trainbatch_size < 2: break
                    print('train batch size too large! change to:', self.trainbatch_size)
                    self.delete_log_model()
                    torch.cuda.empty_cache()
                    torch.cuda.init()
                    torch.cuda.empty_cache()
                    self.trainConfig(param.trainOutput_root, trainDataset, self.trainbatch_size,
                                     testDataset, param.torch_shuffle, param.drop_last, param.tips)
                else:
                    break

    def trainConfig(self, root, trainDataset, trainbatch_size: int = 60, testDataset=None, shuffle=False,
                    drop_last=False, tips=None):  # , testbatch_size=12
        self.trainbatch_size = trainbatch_size
        # self.testbatch_size = testbatch_size
        self.train_loader = DataLoader(dataset=trainDataset, batch_size=trainbatch_size, shuffle=shuffle,
                                       drop_last=drop_last)
        if testDataset is not None:
            self.test_loader = DataLoader(dataset=testDataset, batch_size=trainbatch_size, shuffle=shuffle,
                                          drop_last=drop_last)
        print(self.netname, ':Train batch epoch=', len(self.train_loader), '\tbatch_size=', trainbatch_size)

        self.netPath = os.path.join(root, self.netname)
        if not os.path.isdir(self.netPath):  # 目录不存在就创建
            os.makedirs(self.netPath)
        timestr = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S_{}_')
        self.log_name = timestr + self.log_ispre + 'bc{}'.format(trainbatch_size) + tips
        # files = os.listdir(self.netPath)
        '''记录本次训练配置信息'''
        dataloader_shuffers = str([da for da in self.train_loader.sampler][:10])
        dataset_shuffers = str(trainDataset.shufer_sample)
        # print(dataloader_shuffers,dataset_shuffers)
        self.record_training_log(shuffle=shuffle, drop_last=drop_last, tips=tips,
                                 dataloader_shuffers=dataloader_shuffers, dataset_shuffers=dataset_shuffers)
        pass

    def train(self, train_loader, device, optimizer: optim, start_epoch, stop_epoch, save_frequency,
              startSave_acc=0.8, scheduler=None, online_test=False):

        global test_loss, test_acc
        test_loss, test_acc = '', ''
        net = self.net.to(device)
        net.train()  ## 网络定义为训练模式
        # print(net)
        criterion = nn.CrossEntropyLoss().to(device)  #

        #### -----训练开始-------------------------------------------------------------------------------------------
        print('开始训练：---------------------------------------------------')
        # maxTestacc = 0
        min_val_loss = 0.4
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        '''数据保存位置'''
        train_data_savefile = os.path.join(self.netPath, self.log_name.format('data') + '.csv')
        saveTraindata(train_data_savefile, None, True)  ## 数据文件写入 header

        # for epoch_idx in range(start_epoch,stop_epoch):
        epoch_idx = start_epoch - 1
        while epoch_idx < stop_epoch - 1:
            epoch_idx += 1
            begin = datetime.datetime.now()
            val_loader = []
            train_loss_ = 0.0
            train_acc_ = 0.0
            num_samples = 0
            valIndex = random.sample(range(0, len(train_loader)), int(len(train_loader) / 10))
            for i, dataloader in enumerate(train_loader):
                if i in valIndex:
                    val_loader.append(dataloader)  ### 取出验证集，并跳过用于训练
                    continue
                (inputs, labels) = dataloader
                inputs, labels = inputs.to(device), labels.to(device)
                # print( np.array(inputs).shape, np.array(labels).shape,labels)

                optimizer.zero_grad()
                outputs = net(inputs)
                # print('outputs:',outputs)
                # print('labels:',labels)
                # print(inputs.size(),outputs.size(),labels.size())

                loss = criterion(outputs, labels)  # .to(device)
                loss.backward()
                optimizer.step()
                train_loss_ += float(loss.item())
                pred = outputs.argmax(dim=1)
                train_acc_ += torch.eq(pred, labels).sum().float().item()
                num_samples += len(inputs)
                # torch.cuda.empty_cache()  ## 清理GPU内存

            train_loss.append(train_loss_ / (len(train_loader) - len(val_loader)))
            train_acc.append(train_acc_ / num_samples)

            #### 验证集 -----------------------
            torch.cuda.empty_cache()  ## 清理GPU内存
            val_loss_, val_acc_ = self.test_module(net=net, device=device, test_loader=val_loader)
            torch.cuda.empty_cache()  ## 清理GPU内存
            net.train()
            val_loss.append(val_loss_)
            val_acc.append(val_acc_)

            ## 训练信息保存
            end = datetime.datetime.now()
            lr = self.get_learn_rate(optimizer)
            # headers = ['index', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'spend_time', 'lr']
            train_data = {'index': epoch_idx, 'train_loss': train_loss[-1], 'train_acc': train_acc[-1],
                          'val_loss': val_loss[-1], 'val_acc': val_acc[-1],
                          'test_loss': test_loss, 'test_acc': test_acc,
                          'spend_time': float((end - begin).total_seconds()), 'lr': lr}
            saveTraindata(train_data_savefile, train_data)

            ## -----训练过程可视化-----------------------------------------------------------------------
            # dynamicshow('training'+note,'train result', 200, 'train_loss',train_loss,'train_acc',train_acc)
            # print("- Train: {}/{}---train_Loss: {:.6f}\t train_Acc: {:.6f}\t val_Loss:{}"
            #       "\t val_Acc:{}".format(epoch_idx + 1, stop_epoch, train_loss[-1],
            #                              train_acc[-1], val_loss[-1], val_acc[-1]))
            print("-- Train:", '{idx:' + str(train_data['index']) + ', ',
                  't_loss:{:.6f}, '.format(train_data['train_loss']), 't_acc:{:.6f}, '.format(train_data['train_acc']),
                  'v_loss:{:.6f}, '.format(train_data['val_loss']), 'v_acc:{:.6f}'.format(train_data['val_acc']) + '}',
                  'time:{:.6f},'.format(train_data['spend_time']), 'lr:', train_data['lr'],
                  '  test:[' + str(test_loss) + ', ' + str(test_acc) + ']')

            ## 训练过程动态调节策略 -----------------------------------------------------------
            scheduler.step(val_acc[-1])

            #### -----训练中测试 模型保存----------------------------------------------------------------------
            if val_acc[-1] > startSave_acc or epoch_idx % save_frequency == 0:
                if val_loss[-1] < min_val_loss:  ## 保存验证集损失值最低的模型
                    min_val_loss = val_loss[-1]
                    self.saveModel_onlyOne(net, optimizer, epoch_idx)
            # optimizer.

            ###自动退出
            if np.array(train_loss)[-1:] > 1.2 and len(train_acc) > 6:
                if (max(np.array(train_acc)[-4:]) - min(np.array(train_acc)[-4:])) < 0.01 and \
                        (max(np.array(train_loss)[-4:]) - min(np.array(train_loss)[-4:])) < 0.008:
                    print('\n----------------------------------------------------\n',
                          '--------------- 过拟合退出！！！！ ----------------------')
                    self.delete_log_model()
                    self.exit = True
                    break

            ## 通过文件 Control.py 在线控制 --------------
            # stop_epoch, optimizer, stop_train, save_model, test_model
            control_online = self.control_online(stop_epoch, optimizer)
            stop_epoch = control_online['stop_epoch']
            optimizer = control_online['optimizer']
            if control_online['stop_train']: break
            if control_online['save_model']: self.saveModel_onlyOne(net, optimizer, epoch_idx)
            if control_online['test_model'] or online_test:
                torch.cuda.empty_cache()  ## 清理GPU内存
                test_loss, test_acc = self.test_module(net=net, device=device, test_loader=self.test_loader)
                torch.cuda.empty_cache()  ## 清理GPU内存
            else:
                test_loss, test_acc = '', ''
        if not self.exit:
            self.isremove_all = False
            self.saveModel_onlyOne(net, optimizer, epoch_idx, end=True)
        return epoch_idx + 1
        pass

    def test_module(self, net=None, device=None, test_loader=None, netprint=False):
        criterion = nn.CrossEntropyLoss().to(device)
        if netprint:
            print(net)
        net.eval()
        test_loss_ = 0.0
        test_acc_ = 0.0
        num_samples = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = (inputs).to(device), (labels).to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss_ += float(loss.item())
            # loss.zero_grad()
            pred = outputs.argmax(dim=1)
            test_acc_ += torch.eq(pred, labels).sum().float().item()
            num_samples += len(inputs)
        torch.cuda.empty_cache()  ## 清理GPU内存
        # print("test Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(num_epoch + 1, test_loss_ / len(test_loader),
        #                                                             test_acc_ / len(test_loader.dataset)))
        test_loss = test_loss_ / len(test_loader)
        test_acc = test_acc_ / num_samples

        # if test_acc / len(test_loader.dataset) > 0.98:
        #     break

        # print('Finished testing')
        net.train()
        return test_loss, test_acc

    def model_evaluation(self, modelfile=None, test_loader=None, device=None, **kwargs):
        model_dict = torch.load(modelfile)
        self.net.load_state_dict(model_dict["Net"])
        self.net = self.net.to(device)
        return self.test_module(net=self.net, device=device, test_loader=test_loader, **kwargs)


    def optimizer(self):
        optimizer = optim.SGD(self.net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5 * 1e-4)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6,
                                                               verbose=False)  ## val_acc  , min_lr=1e-10
        return optimizer, scheduler

    def get_learn_rate(self, optimizer):
        res = []
        for group in optimizer.param_groups:
            res.append(group['lr'])
        # for i, param_group in enumerate(optimizer.param_groups):
        #     for j in range(len(param_group)):
        #         res.append(param_group[j]['lr'])
        return res

    def set_learn_rate(self, optimizer, lr=0.01):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
            # for j in range(len(param_group['lr'])):
            #     param_group[j]['lr'] = lr
        return optimizer

    def saveModel_onlyOne(self, net, optimizer, epoch, end=False):
        modulePATH = self.netPath
        if not end:
            filelist = os.listdir(modulePATH)
            for filename in filelist:
                if filename.endswith('.model'):
                    if filename.startswith(self.log_name.format('checkpoint') + '_L'):
                        file = os.path.join(modulePATH, filename)  # 最终参数模型
                        os.remove(file)
                # if filename.startswith(('scriptmodule_{}.pt').format(note)):
                #     file = os.path.join(modulePATH, ('scriptmodule_{}.pt').format(note))
                #     os.remove(file)
            filepath = os.path.join(modulePATH,
                                    self.log_name.format('checkpoint') + '_L_{}.model'.format(epoch))  # 最终参数模型
        else:
            filepath = os.path.join(modulePATH,
                                    self.log_name.format('checkpoint') + '_L_{}_end.model'.format(epoch))  # 最终参数模型
        state = {'Net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, filepath)
        print('torch.save(state)', filepath)

    def continue_training_load_state(self, optimizer, param):
        print('continue trainning ....')
        model_dict = torch.load(param.model_file)
        self.net.load_state_dict(model_dict["Net"])
        if optimizer is not None:
            optimizer.load_state_dict(model_dict['optimizer'])  # model_dict['optimizer']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        epoch = model_dict['epoch']
        return optimizer, epoch

    def control_online(self, stop_epoch, optimizer):
        res = controlManage()
        stop_train = False
        save_model = False
        test_model = False
        if res['epoch'] is not None:
            print('原始 epoch={} 更替为：{}'.format(stop_epoch, res['epoch']))
            stop_epoch = res['epoch']

        if res['lr'] is not None:
            new_lr = res['lr']
            optimizer = self.set_learn_rate(optimizer, new_lr)
        if res['save_model']:
            save_model = True
        if res['test_model']:
            test_model = True

        if res['exit'] is True:
            print('\n----------------------------------------------------\n',
                  '---------------  收到退出指令-->退出  ！！！！ ----------------------')
            stop_train = True
            if res['remove_txt'] is True:

                self.delete_log_model(ismodel=False)
                # shutil.move(txtPath + 'train{}.txt'.format(note), outremovePath)
                # shutil.move(txtPath + 'val{}.txt'.format(note), outremovePath)
                # shutil.move(txtPath + 'test{}.txt'.format(note), outremovePath)
                # os.remove(txtPath + 'train{}.txt'.format(note))
                # os.remove(txtPath + 'val{}.txt'.format(note))
                # os.remove(txtPath + 'test{}.txt'.format(note))
                if res['remove_model'] is True:
                    self.delete_log_model(ismodel=True)
                    self.exit = True
            # print('-----------  清除数据文件完成  移动到文件夹:{} -----------------\n\n'.format(outremovePath))
        # stop_epoch, optimizer, stop_train, save_model, test_model
        res = {'stop_epoch': stop_epoch, 'optimizer': optimizer, 'stop_train': stop_train,
               'save_model': save_model, 'test_model': test_model}
        return res

    def delete_log_model(self, logName=None, ismodel=True, islog=True):
        if logName is None: logName = self.log_name
        print('------  清理文件：  -----------------')  # ,self.netPath, self.log_name
        if ismodel:
            try:
                file1 = logName.format('checkpoint')
                model_list = os.listdir(self.netPath)
                for model in model_list:
                    if model.endswith('.model') and model.startswith(file1):
                        os.remove(os.path.join(self.netPath, model))
                        print('\t', model)
            except:
                pass
            try:
                file2 = logName.format('log') + '.py'
                os.remove(os.path.join(self.netPath, file2))
                print('\t', file2)
            except:
                pass
        if islog:
            try:
                file3 = logName.format('data') + '.csv'
                os.remove(os.path.join(self.netPath, file3))
                print('\t', file3)
            except:
                pass
        pass

    def record_training_log(self, **kwargs):
        file1 = self.log_name.format('log') + '.py'
        file1 = os.path.join(self.netPath, file1)
        log = record2Text()
        if log != '':
            with open(file=file1, mode='a', encoding='utf-8') as f:
                f.write(log)
        self.record_dataloader(file1)
        self.record_netconfig(**kwargs)
        log = self.record_add_log()
        if log != '':
            with open(file=file1, mode='a', encoding='utf-8') as f:
                f.write(log)

    def record_netconfig(self, shuffle=False, drop_last=False, tips=None, dataloader_shuffers='', dataset_shuffers=''):
        file1 = self.log_name.format('log') + '.py'
        file1 = os.path.join(self.netPath, file1)
        with open(file=file1, mode='a', encoding='utf-8') as f:
            f.write("'''" + self.netname + "'''\n")
            f.write('trainbatch_size = ' + str(self.trainbatch_size) + '\n')
            f.write('testbatch_size = ' + str(self.testbatch_size) + '\n')
            f.write('shuffle = ' + str(shuffle) + '\n')
            f.write('drop_last = ' + str(drop_last) + '\n')
            f.write('tips = ' + str(tips) + '\n')
            f.write('\ndataloader_shuffers = ' + str(dataloader_shuffers) + '\n')
            f.write('dataset_shuffers = ' + str(dataset_shuffers) + '\n')
            f.write("\n\n## -- train_base_class: [{}] --\n\n".format(self._class_file_))
            with open(self._class_file_, mode='r', encoding='utf-8') as f2:
                # print('__file__:', self._class_file_)
                start_p = False
                lines = f2.readlines()
                for line in lines:
                    if 'def optimizer(self)' in line: start_p = True
                    if 'return optimizer' in line:
                        f.write(line)
                        break
                    if start_p:
                        f.write(line)

        pass

    def record_dataloader(self, file1):
        with open(file=file1, mode='a', encoding='utf-8') as f:
            f.write("\n'''\ntrainDataLoader.dataSet = " + str(self.train_loader.dataset.__class__.__name__) + "\n'''\n")

    def record_add_log(self):  ##->str
        return ''

    def train_minibatch(self, train_loader, device, optimizer: optim, start_epoch, stop_epoch, save_frequency,
                        startSave_acc=0.8, scheduler=None, mini_batch=128, drop_last=False, online_test=False):

        global test_loss, test_acc
        test_loss, test_acc = '', ''
        net = self.net.to(device)
        net.train()  ## 网络定义为训练模式
        # print(net)
        criterion = nn.CrossEntropyLoss().to(device)  #

        #### -----训练开始-------------------------------------------------------------------------------------------
        print('开始训练：------------------------------------ mini_batch:', mini_batch)
        min_val_loss = 0.4
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        '''数据保存位置'''
        train_data_savefile = os.path.join(self.netPath, self.log_name.format('data') + '.csv')
        saveTraindata(train_data_savefile, None, True)  ## 数据文件写入 header
        optimizer.zero_grad()
        '''计算 drop_last 的控制点 no_drop_num '''
        loaderIndex = range(0, len(train_loader))
        train_loader_num = int(len(train_loader) - int(len(train_loader) / 10))
        ## 得到trainIndex的必定训练长度
        no_drop_num = int(
            train_loader_num * self.trainbatch_size / mini_batch) * mini_batch / self.trainbatch_size

        # print('train_loader_num:', train_loader_num, 'no_drop_num:', no_drop_num)

        # for epoch_idx in range(start_epoch,stop_epoch):
        epoch_idx = start_epoch - 1
        while epoch_idx < stop_epoch - 1:
            epoch_idx += 1
            begin = datetime.datetime.now()
            train_loss_ = 0.0
            train_acc_ = 0.0
            val_loader = []
            num_samples = 0

            accumulate_iter = int(mini_batch / self.trainbatch_size)  # 为达到miniBatch，需要累积的次数
            trainIndex = np.array(random.sample(loaderIndex, len(train_loader) - int(len(train_loader) / 10)))

            # for i, data_loader in enumerate(train_loader):
            #     if i not in trainIndex:
            #         val_loader.append(data_loader)
            #         continue
            #     inputs, labels = data_loader
            #     inputs, labels = inputs.to(device), labels.to(device)
            #     outputs = net(inputs)
            #     loss = criterion(outputs, labels) / accumulate_iter  # .to(device)
            #     loss.backward()
            #
            #     train_loss_ += float(loss.item())
            #     print( 'train_loss_:', train_loss_, loss.item(), accumulate_iter)
            #     pred = outputs.argmax(dim=1)
            #     train_acc_ += torch.eq(pred, labels).sum().float().item()
            #     num_samples += len(inputs)
            #
            #     if ((i + 1) <= no_drop_num and (i + 1) % accumulate_iter == 0) or \
            #             ((i + 1) > no_drop_num and (i - no_drop_num) % no_drop_num == 0):
            #         optimizer.step()  # 根据梯度更新网络参数
            #         optimizer.zero_grad()  # 梯度清零
            #     if (i + 1) == no_drop_num:
            #         if not drop_last:
            #             accumulate_iter = len(trainIndex) - no_drop_num
            #         else:
            #             break
            i_train = -1
            for i, data_loader in enumerate(train_loader):
                if i not in trainIndex:
                    val_loader.append(data_loader)
                    continue
                i_train = i_train + 1
                inputs, labels = data_loader
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels) / accumulate_iter  # .to(device)
                loss.backward()

                train_loss_ += float(loss.item())
                # print('train_loss_:', train_loss_, loss.item(), accumulate_iter)
                pred = outputs.argmax(dim=1)
                train_acc_ += torch.eq(pred, labels).sum().float().item()
                num_samples += len(inputs)

                if ((i_train + 1) <= no_drop_num and (i_train + 1) % accumulate_iter == 0) or \
                        ((i_train + 1) > no_drop_num and (i_train - no_drop_num) % no_drop_num == 0):
                    optimizer.step()  # 根据梯度更新网络参数
                    optimizer.zero_grad()  # 梯度清零
                if (i_train + 1) == no_drop_num:
                    if not drop_last:
                        accumulate_iter = len(trainIndex) - no_drop_num
                    else:
                        break

                # torch.cuda.empty_cache()  ## 清理GPU内存

            if drop_last:
                train_loss.append(train_loss_ / no_drop_num)
                # train_acc.append(train_acc_ / (no_drop_num*self.trainbatch_size))
            else:
                train_loss.append(train_loss_ / len(trainIndex))
            train_acc.append(train_acc_ / num_samples)  ## (len(trainIndex)*self.trainbatch_size))

            #### 验证集 -----------------------
            torch.cuda.empty_cache()  ## 清理GPU内存
            val_loss_, val_acc_ = self.test_module(net=net, device=device, test_loader=val_loader)
            torch.cuda.empty_cache()  ## 清理GPU内存
            net.train()
            val_loss.append(val_loss_)
            val_acc.append(val_acc_)

            ## 训练信息保存
            end = datetime.datetime.now()
            lr = self.get_learn_rate(optimizer)
            # headers = ['index', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'spend_time', 'lr']
            train_data = {'index': epoch_idx, 'train_loss': train_loss[-1], 'train_acc': train_acc[-1],
                          'val_loss': val_loss[-1], 'val_acc': val_acc[-1],
                          'test_loss': test_loss, 'test_acc': test_acc,
                          'spend_time': float((end - begin).total_seconds()), 'lr': lr}
            saveTraindata(train_data_savefile, train_data)

            ## -----训练过程可视化-----------------------------------------------------------------------
            # dynamicshow('training'+note,'train result', 200, 'train_loss',train_loss,'train_acc',train_acc)
            # print("- Train: {}/{}---train_Loss: {:.6f}\t train_Acc: {:.6f}\t val_Loss:{}"
            #       "\t val_Acc:{}".format(epoch_idx + 1, stop_epoch, train_loss[-1],
            #                              train_acc[-1], val_loss[-1], val_acc[-1]))
            print("-- Train:", '{idx:' + str(train_data['index']) + ', ',
                  't_loss:{:.6f}, '.format(train_data['train_loss']), 't_acc:{:.6f}, '.format(train_data['train_acc']),
                  'v_loss:{:.6f}, '.format(train_data['val_loss']), 'v_acc:{:.6f}'.format(train_data['val_acc']) + '}',
                  'time:{:.6f},'.format(train_data['spend_time']), 'lr:', train_data['lr'],
                  '  test:[' + str(test_loss) + ', ' + str(test_acc) + ']')

            ## 训练过程动态调节策略 -----------------------------------------------------------
            scheduler.step(val_acc[-1])

            #### -----训练中测试 模型保存----------------------------------------------------------------------
            if val_acc[-1] > startSave_acc or epoch_idx % save_frequency == 0:
                if val_loss[-1] < min_val_loss:  ## 保存验证集损失值最低的模型
                    min_val_loss = val_loss[-1]
                    self.saveModel_onlyOne(net, optimizer, epoch_idx)
            # optimizer.

            ###自动退出
            if np.array(train_loss)[-1:] > 1.2 and len(train_acc) > 6:
                if (max(np.array(train_acc)[-4:]) - min(np.array(train_acc)[-4:])) < 0.01 and \
                        (max(np.array(train_loss)[-4:]) - min(np.array(train_loss)[-4:])) < 0.008:
                    print('\n----------------------------------------------------\n',
                          '--------------- 过拟合退出！！！！ ----------------------')
                    self.delete_log_model()
                    self.exit = True
                    break

            ## 通过文件 Control.py 在线控制 --------------
            # stop_epoch, optimizer, stop_train, save_model, test_model
            control_online = self.control_online(stop_epoch, optimizer)
            stop_epoch = control_online['stop_epoch']
            optimizer = control_online['optimizer']
            if control_online['stop_train']: break
            if control_online['save_model']: self.saveModel_onlyOne(net, optimizer, epoch_idx)
            if control_online['test_model'] or online_test:
                torch.cuda.empty_cache()  ## 清理GPU内存
                test_loss, test_acc = self.test_module(net=net, device=device, test_loader=self.test_loader)
                torch.cuda.empty_cache()  ## 清理GPU内存
            else:
                test_loss, test_acc = '', ''
        if not self.exit:
            self.isremove_all = False
            self.saveModel_onlyOne(net, optimizer, epoch_idx, end=True)
        return epoch_idx + 1
        pass

    def __del__(self):
        print('{}类退出'.format(self.netname))  # \033[31m  \033[0m
        if self.isremove_all:
            self.delete_log_model()
