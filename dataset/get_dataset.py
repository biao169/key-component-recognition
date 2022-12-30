import torch
import torch.utils.data
from PIL import Image
import os
import random
import numpy as np
import cv2 as cv

"""数据随机时，样本平衡"""
def dataShuffer(data_label_list):
    """ 将原本按 AABBBCC……排布变成ABCABC……"""
    labels = []
    idx_eachClass = []
    '''获取各类标签的索引'''
    for i, words in enumerate(data_label_list):
        if words[1] not in labels:
            labels.append(words[1])  # 标签缓存
            da = [i]
            idx_eachClass.append(da)
        else:
            idx = labels.index(words[1])
            idx_eachClass[idx].append(i)
    '''按顺序，平衡类别'''  ## len(labels)为周期
    max_len = 0
    class_idx = random.sample(range(0, len(idx_eachClass)), len(idx_eachClass))
    for idx in class_idx:
        random.shuffle(idx_eachClass[idx])
        lens = len(idx_eachClass[idx])
        if max_len < lens:
            max_len = lens
    imagine_index = []
    for i in range(max_len):
        class_idx = random.sample(range(0, len(idx_eachClass)), len(idx_eachClass))
        for idx in class_idx:
            if i < len(idx_eachClass[idx]):  # 保证索引在该类的数量下
                imagine_index.append(idx_eachClass[idx][i])
    return imagine_index


"""图片样本数据加倍：翻转，旋转, 可同时翻倍标签"""
def imgmultiply(imgdatalist, num_multiply: int = 0):
    """ 将图片按顺序，根据增强倍数，翻转，旋转等分角度"""
    if num_multiply <= 1:  # 二重保护，倍数为<=1，则直接返回原始数据
        return imgdatalist
    resultlist = []  # 结果保存
    for data in imgdatalist:  # 依次加倍每个图
        try:
            row = data.shape[0]
            col = data.shape[1]  ##获取图片shape，当实参是标签数据时报错
            #### 优先
            resultlist.append(data)  # 第一个是原图
            if num_multiply - 2 >= 0:  ##num_multiply=1
                resultlist.append(cv.flip(data, 0))  # X翻转 第二个
            if num_multiply - 3 >= 0:
                resultlist.append(cv.flip(data, -1))  # （任意负数） X-Y翻转 第三个
            if num_multiply - 4 >= 0:
                resultlist.append(cv.flip(data, 1))  # （任意正数）Y翻转  第四个

            num = num_multiply - 3
            if num - 1 > 0:  # 当倍数大于4时，则，通过翻转角度扩充样本
                for i in random.sample(range(1, num), num - 1):  # 随机产生1-num-1的值
                    randangle = i * int((360 / (num)))  # 将360°等分，避开0=360的重复
                    M = cv.getRotationMatrix2D((row / 2, col / 2), randangle, 0.9)  # 生成旋转矩阵，缩放0.9
                    imgdst = cv.warpAffine(data, M, (row, col))  # 开始旋转变换
                    resultlist.append(imgdst)  # 保存图片
        except Exception as e:  # 标签list进入此
            for i in range(num_multiply):  # 直接复制扩充
                resultlist.append(data)
    return resultlist


'''使用 PIL 读取图片数据，使用 torch 自带transform来数据增强'''
class Dataset_plt(torch.utils.data.Dataset):
    """ 可控使用在线读取数据，测试GPU情况 """
    def __init__(self, imagePath, labelfile, transform, shuffer=True, imgSize=None, online=True, **kwargs):
        super(Dataset_plt, self).__init__()
        self.root = imagePath
        self.imagePath_label, labelList = self.getfile_and_label(os.path.join(imagePath, labelfile))
        self.shuffer = shuffer
        self.class_num = {'num_class': len(labelList.items()), 'detail': labelList, 'all samples:': np.array(list(labelList.values())).sum()}
        self.transform = transform
        self.online = online
        self.imgSize = imgSize
        if online is False:
            self.imagedata = self.readImages()
        if shuffer:
            self.img_index = dataShuffer(self.imagePath_label)  ## 保持每个周期中，样本都平衡
        else:
            self.img_index = np.arange(len(self.imagePath_label))
        self.shufer_sample = self.img_index[:10]

    def __getitem__(self, idx):  ## 不/在线读取数据
        item = self.img_index[idx]
        if self.online:
            file, label = self.imagePath_label[item]
            img = self.read_img(os.path.join(self.root, file))  # 把图像转成RGB
        else:
            img, label = self.imagedata[item], self.imagePath_label[item][1]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(int(label), dtype=torch.long)
        print(idx, '===', img.size(), label)
        return img, label

    def __len__(self):
        return len(self.imagePath_label)

    def readImages(self):
        images = []
        for item in range(len(self.imagePath_label)):
            file, label = self.imagePath_label[item]
            img = self.read_img(os.path.join(self.root, file))
            images.append(img)
        return images

    def read_img(self, file):
        img = Image.open(file).convert('RGB')  # 把图像转成RGB
        # img = img.resize(self.imgSize)
        return img


    def getfile_and_label(self, file):
        cal_eachClass = {}
        imgs_path_label = []
        labels = []
        with open(file, 'r') as file_txt:
            for line in file_txt:
                # print(line)
                line = line.rstrip()  ##删除字符串末尾的字符，默认为空格
                words = line.split('\t')[:2]  ## 根据分隔符，切分 !!! 括号内不能有空格（不能以空格作为分隔符）
                imgs_path_label.append([words[0], words[1]])
                if words[1] not in labels:
                    labels.append(words[1])
                    newClass = {words[1]: 1}
                    cal_eachClass = dict(cal_eachClass, **newClass)
                else:
                    cal_eachClass[words[1]] = int(cal_eachClass[words[1]]) + 1
        return imgs_path_label, cal_eachClass


'''使用 opencv 读取图片数据，使用 自定义imgmultiply 来数据增强'''
class Dataset_cv(torch.utils.data.Dataset):
    """ """
    def __init__(self, imagePath, labelfile, transform, shuffer=True, imgSize=None, online=True, enhanceNum=1, **kwargs):
        super(Dataset_cv, self).__init__( )
        self.root = imagePath
        self.imagePath_label, labelList = self.getfile_and_label(os.path.join(imagePath, labelfile))
        # self.transform = transform
        self.shuffer = shuffer
        if enhanceNum > 1: online = False
        self.online = online
        self.imgSize = imgSize

        if online is False:
            self.imagedata = self.readImages()

        if enhanceNum > 1:
            self.imagedata = imgmultiply(imgdatalist=self.imagedata, num_multiply=enhanceNum)
            self.imagePath_label = imgmultiply(imgdatalist=self.imagePath_label, num_multiply=enhanceNum)
            labelList2 = {}
            for k in labelList.keys(): labelList2[k] = labelList[k]*enhanceNum
            labelList = labelList2

        self.class_num = {'num_class': len(labelList.items()), 'detail': labelList, 'all samples:': np.array(list(labelList.values())).sum()}

        if shuffer:
            self.img_index = dataShuffer(self.imagePath_label)  ## 保持每个周期中，样本都平衡
        else:
            self.img_index = np.arange(len(self.imagePath_label)*enhanceNum)
        self.shufer_sample = self.img_index[:10]

    def __getitem__(self, idx):  ## 不/在线读取数据
        item = self.img_index[idx]
        if self.online:
            file, label = self.imagePath_label[item]
            img = self.read_img(os.path.join(self.root, file))
        else:
            img, label = self.imagedata[item], self.imagePath_label[item][1]

        img = torch.from_numpy(img.transpose((2, 0, 1)))  # np.array
        img = img.to(torch.float)
        img = img.div(255.0)

        label = torch.tensor(int(label), dtype=torch.long)
        # print(idx,'===', img.size() , label)
        return img, label

    def read_img(self, file):
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, self.imgSize)
        # img = np.array(img, dtype=np.uint8)
        return img

    def __len__(self):
        return len(self.imagePath_label)

    def readImages(self):
        images = []
        for item in range(len(self.imagePath_label)):
            file, label = self.imagePath_label[item]
            img = self.read_img(os.path.join(self.root, file))
            images.append(img)
        return images

    def getfile_and_label(self, file):
        cal_eachClass = {}
        imgs_path_label = []
        labels = []
        with open(file, 'r') as file_txt:
            for line in file_txt:
                # print(line)
                line = line.rstrip()  ##删除字符串末尾的字符，默认为空格
                words = line.split('\t')[:2]  ## 根据分隔符，切分 !!! 括号内不能有空格（不能以空格作为分隔符）
                imgs_path_label.append([words[0], words[1]])
                if words[1] not in labels:
                    labels.append(words[1])
                    newClass = {words[1]: 1}
                    cal_eachClass = dict(cal_eachClass, **newClass)
                else:
                    cal_eachClass[words[1]] = int(cal_eachClass[words[1]]) + 1
        return imgs_path_label, cal_eachClass

from utils_tool.ImageProcessing_Set import imageConcentration
'''使用 opencv 读取图片数据，使用 自定义imgmultiply 来数据增强，Concentration  '''
class Dataset_Concentration(Dataset_cv):
    """ """
    def __init__(self, imagePath, labelfile, transform, shuffer=True, imgSize=None, online=True, enhanceNum=1, k=1.2, **kwargs):
        super(Dataset_Concentration, self).__init__( imagePath, labelfile, transform, shuffer, imgSize, online, enhanceNum, **kwargs)
        self.k = k
        pass

    def read_img(self, file):
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = cv.resize(img, self.imgSize)
        # img = np.array(img, dtype=np.uint8)
        return img

    def __getitem__(self, idx):  ## 不/在线读取数据
        item = self.img_index[idx]
        if self.online:
            file, label = self.imagePath_label[item]
            img = self.read_img(os.path.join(self.root, file))
        else:
            img, label = self.imagedata[item], self.imagePath_label[item][1]
        img = imageConcentration(img, k=self.k, ishow=False)
        img = cv.resize(img, self.imgSize)
        img = torch.from_numpy(img.transpose((2, 0, 1)))  # np.array
        img = img.to(torch.float)
        img = img.div(255.0)

        label = torch.tensor(int(label), dtype=torch.long)
        # print(idx,'===', img.size() , label)
        return img, label