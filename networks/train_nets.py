import datetime
import os
import random

import torch
from torch import nn, optim

from utils_tool.log_utils import Summary_Log, Visual_Model_Predict



def func_param_to_dict(**kwargs):
    return kwargs

class Train_base:
    def __init__(self):
        self.epochNum = 0
        self.resultPath = os.getcwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ("cpu") #
        self.modelPath = os.path.join(self.resultPath, 'model')
        self.startTime = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        self.train_dataLoader = None  # classmethod
        self.test_dataLoader = None  # classmethod
        self.summary_log = None  ## Summary_Log(path=self.resultPath, headers=None, write_csv=True,tm_str=self.startTime, save_log=True)

        self.net = nn.Module()
        self.opti_log = {}
        self.config:{} = None
        pass

    def config_make(self):
        args = {
            'device': str(self.device),
            'resultPath': self.resultPath,
            'epochNum': self.epochNum,
        }
        if self.train_dataLoader is not None:
            try:
                path = self.train_dataLoader.dataset.config_make()
            except:
                try: path = self.train_dataLoader.dataset.root
                except: path = self.train_dataLoader.dataset.path
            arg2 = {'train_loader': {
                'miniBatch': self.train_dataLoader.batch_size,
                'dataset': path
            }}
            args = {**args, **arg2}

        if self.test_dataLoader is not None:
            try:
                path = self.test_dataLoader.dataset.config_make()
            except:
                try: path = self.test_dataLoader.dataset.root
                except: path = self.test_dataLoader.dataset.path
            arg2 = {'test_loader': {
                'miniBatch': self.test_dataLoader.batch_size,
                'dataset': path
            }}
            args = {**args, **arg2}

        try:
            args = {**args, **self.opti_log}
        except: pass
        return args
        pass

    def training_init_save(self):
        try:
            self.opti_log['network'] = {'name': str(self.net.__class__),
                                        'file path': self.net.___file__,
                                        'log': self.net.log}

            args = self.config_make()
            for key in self.config:
                if key in args.keys(): continue
                args[key] = self.config[key]
            self.summary_log.save_config(args)
            os.makedirs(self.modelPath, exist_ok=True)
            print(f'[base]: train batchSize={self.train_dataLoader.batch_size}, \ttrainLoader num={len(self.train_dataLoader)}')
        except: pass

    def optimizer(self, net=None,  opt_name:str='Adam', kwargs:{}=None):
        # lr = 1e-4, momentum = 0.9, weight_decay = 5 * 1e-4
        # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
        if net is None: net = self.net
        if opt_name.lower() == 'sgd':
            optimizer = optim.SGD(net.parameters(), **kwargs)
        else:
            optimizer = optim.Adam(net.parameters(), **kwargs)
        self.opti_log['optimizer'] = {opt_name: kwargs}
        return optimizer

    def optimizer_multi_nets_parameters(self, params=None, opt_name: str = 'Adam', kwargs: {} = None):
        # lr = 1e-4, momentum = 0.9, weight_decay = 5 * 1e-4
        # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False  params={'params': param_groups}
        """
        :param params: 可以是多个网络参数，用list['params'：parameters]组合
        :param opt_name: SGD // Adam
        :param kwargs:  {lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,]
        :return: optimizer
        """
        """ params = [{'params': model.parameters()}, {'params': lossnet.parameters(), 'lr': 1e-4}]  """
        if params is None: params = self.net.parameters()
        if opt_name.lower() == 'sgd':
            optimizer = optim.SGD(params, **kwargs)
        else:
            optimizer = optim.Adam(params, **kwargs)
        self.opti_log['optimizer'] = {opt_name: kwargs}
        return optimizer

    def scheduler(self, optimizer, sched_name: str = 'ReduceLROnPlateau', kwargs:{} = None):
        if sched_name.lower()=='ReduceLROnPlateau'.lower():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
            #                                                        eps=1e-10, cooldown=5,
            #                                                        verbose=False)  ## val_acc  , min_lr=1e-10
            # kwargs = func_param_to_dict(mode='min', factor = 0.5, patience = 30, eps = 1e-10, cooldown = 5, verbose = False)

        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

        self.opti_log['scheduler'] = {**kwargs}
        return scheduler

    def get_learn_rate(self, optimizer):
        res = []
        for group in optimizer.param_groups:
            res.append(group['lr'])
        return res

    def set_learn_rate(self, optimizer, lr=0.01):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return optimizer

    def saveModel_onlyOne(self, net=None, epoch=0, name=''):
        if not self.summary_log.save_log: return
        modelPath = self.modelPath
        name0= name
        if net is None: net= self.net
        if name =='':
            new_name = 'checkpoint_{}_epo[{}].model'.format(self.startTime, epoch)
            rm_name = '].model'   ##  避免误删
        else:
            new_name = 'checkpoint_{}_epo[{}]_{}.model'.format(self.startTime, epoch, name)
            rm_name = '_{}.model'.format(name)  ##  避免误删

        filelist = os.listdir(modelPath)
        for filename in filelist:
            if filename.endswith('.model'):
                if filename.startswith('checkpoint') and rm_name in filename and self.startTime in filename:
                    file = os.path.join(modelPath, filename)  # 最终参数模型
                    os.remove(file)
                    # print('saveModel_onlyOne:','remove', file)
            # if filename.startswith(('scriptmodule_{}.pt').format(note)):
            #     file = os.path.join(modelPath, ('scriptmodule_{}.pt').format(note))
            #     os.remove(file)
        filepath = os.path.join(modelPath, new_name)  # 最终参数模型
        state = {'Net': net.state_dict(), 'epoch': epoch}
        torch.save(state, filepath)
        print(f'\t[base]: --- torch.save [{name0}] model:', filepath)

    def load_model_weight(self, net, name='', **kwargs):  # param
        files = os.listdir(self.modelPath)
        files.reverse()
        for f in files:
            if str(f).endswith('.model') and name in str(f):
                model_file = os.path.join(self.modelPath, f)
                print('\t[base]: loading model weight:', model_file)
                model_dict = torch.load(model_file)  # param.
                net.load_state_dict(model_dict["Net"])
                epoch = model_dict['epoch']
                return net, epoch
        print('\t[base]: load model weight: [fail]', self.modelPath, files)
        return net, 0

    def load_model_weight_file(self, net, file='', **kwargs):  # param
        model_file = os.path.join(self.modelPath, file)
        print('\t[base]: loading model weight:', model_file)
        model_dict = torch.load(model_file)  # param.
        net.load_state_dict(model_dict["Net"])
        epoch = model_dict['epoch']
        return net, epoch

    def load_model_weight_auto(self, net, **kwargs):
        if 'file' in kwargs:
            net, epoch = self.load_model_weight_file(net, file=kwargs['file'])
        else:
            net, epoch = self.load_model_weight(net, **kwargs)
        return net, epoch

class Train(Train_base):
    def __init__(self, net:nn.Module, train_dataLoader, test_dataLoader=None, num_classes=10, config:{}=None,  **kwargs):
        super(Train, self).__init__()
        self.train_dataLoader = train_dataLoader
        self.test_dataLoader = test_dataLoader
        self.net = net.to(self.device)
        ''' record the address of train class file'''
        self.opti_log['train way file'] = os.path.abspath(__file__)
        ''' configuration class init. '''
        self.config = config
        self.resultPath = config['resultPath']
        self.epochNum = config['epochNum']
        self.modelPath = os.path.join(self.resultPath, 'model')
        ''' set log mode '''
        log_dict = {'train':['loss', 'acc'],'val':['loss', 'acc'], 'test':['loss', 'acc']}
        self.summary_log = Summary_Log(path=self.resultPath, headers=log_dict,
                                       tm_str=self.startTime, **kwargs)
        ''' set loss function '''
        self.loss_fn =  nn.CrossEntropyLoss().to(self.device) ## nn.MSELoss().to(self.device)
        pass

    def training_mode(self, pretrain=False, datasetName='crwu', **kwargs):
        start_epoch = 0
        if pretrain:
            self.net, start_epoch = self.load_model_weight_auto(self.net, **kwargs)
        device = self.device
        net = self.net.to(device)

        optimizer = self.optimizer(net=self.net,
                                   opt_name=str(*self.config['optimizer'].keys()),
                                   kwargs=self.config['optimizer'][str(*self.config['optimizer'].keys())]
                                   )

        scheduler = self.scheduler(optimizer=optimizer,
                                   sched_name=str(*self.config['scheduler'].keys()),
                                   kwargs=self.config['scheduler'][str(*self.config['scheduler'].keys())]
                                   )

        """ training system initialization before start training (only one step in advance)"""
        self.training_init_save()
        print('[train]: ================ starting train ========================')
        """ start training """
        # with torch.autograd.set_detect_anomaly(True):
        for idx_epo in range(start_epoch, self.epochNum):
            net.train()
            acc_num = 0
            train_loss = 0
            mini_batch_num = 0
            train_loader = self.train_dataLoader
            val_loader = []
            valIndex = random.sample(range(0, len(train_loader)), int(len(train_loader) / 10))
            for i, dataloader in enumerate(train_loader):
                if i in valIndex:
                    val_loader.append(dataloader)  ### 取出验证集，并跳过用于训练
                    continue
                (inputs, labels) = dataloader
                inputs, label = inputs.to(device), labels.to(device)
                ''' Design program '''
                output = net(inputs)

                loss = self.loss_fn(output, label)  #.squeeze(dim=1)
                pred = output.argmax(dim=1)
                acc_num += torch.eq(pred, label).sum().float().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                mini_batch_num += len(label)

            torch.cuda.empty_cache()
            train_loss = train_loss / (len(self.train_dataLoader)-len(val_loader))
            train_acc = acc_num / mini_batch_num   ## len(self.train_dataLoader.dataset)

            """ val """
            result_val = self.test_model(net, val_loader, pretrain=False, datasetName=datasetName)

            ''' dynamic adjust lr '''
            try: scheduler.step(result_val['loss'], epoch=None)  ##
            except: scheduler.step(epoch=None)  #  StepLR

            ''' record log ''' # result must the same as the log_dict(above)!
            result = {'loss': train_loss, 'acc': train_acc}
            self.summary_log.add_scalars('train', result, idx_epo, tolerant=True)
            self.summary_log.add_scalars('val', result_val, idx_epo, tolerant=True)

            ''' test model online '''
            if (idx_epo+1) % 1 == 0:
                result = self.test_model(net, self.test_dataLoader, pretrain=False, datasetName=datasetName)
                self.summary_log.add_scalars('test', result, idx_epo, tolerant=True)

            ''' 保存模型 '''
            if train_loss < self.summary_log.getMin('train', 'loss', rm_last=True):
                self.saveModel_onlyOne(self.net, idx_epo, 'best')
            if (idx_epo+1)%5==0:
                self.saveModel_onlyOne(self.net, idx_epo, name='')
        pass

    def test_model(self, net=None, dataLoader=None, pretrain=False, datasetName='crwu', **kwargs):
        if net is None: net =self.net
        if pretrain:
            net, epoch = self.load_model_weight_auto(net, **kwargs)
            net = net.to(self.device)
        if dataLoader is None: dataLoader = self.test_dataLoader

        if net is None or dataLoader is None:
            res = {}
            for k in self.summary_log.headers['test']:
                res[k] = 0
            return res
        device = self.device
        torch.cuda.empty_cache()
        net.eval()
        acc_num = 0
        train_loss = 0
        mini_batch_num = 0
        for i, dataloader in enumerate(dataLoader):
            (inputs, labels) = dataloader
            inputs, label = inputs.to(device), labels.to(device)
            ''' Design program '''
            output = net(inputs)

            loss = self.loss_fn(output, label)  # .squeeze(dim=1)
            pred = output.argmax(dim=1)
            acc_num += torch.eq(pred, label).sum().float().item()
            train_loss += float(loss.item())
            mini_batch_num += len(label)

        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)
        train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)
        return {'loss': train_loss, 'acc': train_acc}

    def test_model_visualization(self, net=None, dataLoader=None, pretrain=False, datasetName='crwu', **kwargs):
        if net is None: net = self.net
        if pretrain:
            net, epoch = self.load_model_weight_auto(net, **kwargs)
            net = net.to(self.device)
        if dataLoader is None: dataLoader = self.test_dataLoader

        if net is None or dataLoader is None:
            res = {}
            for k in self.summary_log.headers['test']:
                res[k] = 0
            return res
        """ Visual """
        vs = Visual_Model_Predict()

        device = self.device
        torch.cuda.empty_cache()
        net.eval()
        acc_num = 0
        train_loss = 0
        mini_batch_num = 0
        for i, dataloader in enumerate(dataLoader):
            (inputs, labels) = dataloader
            inputs, label = inputs.to(device), labels.to(device)
            ''' Design program '''
            output = net(inputs)

            loss = self.loss_fn(output, label)
            pred = output.argmax(dim=1)
            acc_num += torch.eq(pred, label).sum().float().item()
            mini_batch_num += len(label)

            train_loss += float(loss.item())

            vs.add_data_series(data={'label': label.detach().cpu().numpy(), 'predict': pred.detach().cpu().numpy()})
            # print({'label': rate.detach().cpu().numpy(), 'predict': conf.detach().cpu().numpy()})
            # break
        torch.cuda.empty_cache()
        train_loss = train_loss / len(dataLoader)
        train_acc = acc_num / mini_batch_num  ## len(self.train_dataLoader.dataset)

        ''' record log '''  # result must the same as the log_dict(above)!
        result = {'loss': train_loss, 'acc': train_acc}
        print('test model：', result)
        vs.draw_figure_matrix(keys=['label', 'predict'])
        return result




