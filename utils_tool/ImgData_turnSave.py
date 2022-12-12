__author__ = '谭劲标'

### 本文件包含：
##  img_division——切割扁/高瘦型图片(长宽比例ratio阀值可控)；不分割时，返回原图或resize后的图（仅1张）
##                需要分割时，一定需要resize，并返回3张图片（原图+2张分割图）
##  pathdata_delete——清空/删除文件夹（oriDelete=True则删除到path目录，=False则清空path内的文件）

##  img_changePath——将路径文件夹内的子文件夹中的图片复制到其他文件(包含子文件夹)，文件名按序命名！
##                  该过程可以resize，png->jpg
## 输出目录不存在，一律自动创建
##  imgDivide_TrainPredict——将目录内的图片文件，可只读（或复制到新目录内），并分类成训练集和测试集集
##                          返回每类训练集和测试集，剩余未分类集 的样本数量。
## 输出目录不存在，一律自动创建
##  read_ImgDataBase——读取目录文件内的图片数据，每个同子文件夹内的图片问一类。返回img和Label的list型

import numpy as np
import cv2 as cv
import os
import random
import zipfile

# def print(*data0):
#     f = open(outprintTxt,'a')
#     for data in data0:
#         if type(data) == type('s'):
#             f.write(data)
#         else:
#             f.write(str(data))
#         f.write(' ')
#     f.write('\n')
#     f.close()

# from PIL import Image
# import matplotlib.pyplot as plt

###-----------------------------------------------------------------------------------------------
""" 图像滤波操作 // 针对图片列表list """
def imgfilter_list( imgarr):
    imglist = []
    for img in imgarr:
        # cv.imshow('jg', img)
        img = cv.blur(img, (3, 3),(-1,-1))  ## #中值滤波
        img = cv.GaussianBlur(img, (5, 5), 10)  # 高斯滤波

        imglist.append(img)
        # cv.waitKey(200)
    imglist = np.array(imglist)
    return imglist
""" 图像滤波操作 // 针对单张图片 """
def imgfilter( img):
    img = cv.blur(img, (3, 3),(-1,-1))  ## #中值滤波
    img = cv.GaussianBlur(img, (5, 5), 10)  # 高斯滤波
    return img

###-----------------------------------------------------------------------------------------------
"""样本数据加倍 // 针对图片列表list 和 标签列表list，或者单张图片 """
def imgmultiply(imgdata,num:int=0,isList:bool=False):
    resultlist = []
    if num <= 1:
        return imgdata
    elif isList:
        num = num-1  #保留原图

    angle2 = [45, 90, 135, 180, 225, 270, 315]
    angle = ['x', 'y', 'k']
    if isList:
        for data in imgdata:

            try:  ##是图片shape就能运行
                row, col = data.shape[:2]
                if num < 3:
                    num_change = num
                else:
                    num_change = 3
                if isList:
                    resultlist.append(data)  # 保留原图
                for i in random.sample(range(0, int(len(angle))), num_change):
                    if angle[i] == 'x':
                        resultlist.append(cv.flip(data, 0))  # X翻转
                    elif angle[i] == 'y':
                        resultlist.append(cv.flip(data, 1))  # （任意正数）Y翻转
                    elif angle[i] == 'k':
                        resultlist.append(cv.flip(data, -1))  # # （任意负数） X-Y翻转
                if num > 3:
                    i = 0
                    for da in random.sample(range(0, int(len(angle2))), num-3):
                        M = cv.getRotationMatrix2D((row/2, col/2), int(angle2[da]), 1)  # 生成旋转矩阵，缩放0.9
                        imgdst = cv.warpAffine(data, M, (int(row ), int(col )), cv.INTER_LINEAR,
                                               borderValue=10)  # cv.INTER_CUBIC, borderValue= 开始旋转变换
                        resultlist.append(imgdst)
            except Exception as e: # 标签list进入此
                for i in range(num+1):
                    resultlist.append(data)
    else:
        if num < 3:
            num_change = num
        else:
            num_change = 3
        if isList:
            resultlist.append(imgdata)  # 保留原图
        for i in random.sample(range(0, int(len(angle))), num_change):
            if angle[i] == 'x':
                resultlist.append(cv.flip(imgdata, 0))  # X翻转
            elif angle[i] == 'y':
                resultlist.append(cv.flip(imgdata, 1))  # （任意正数）Y翻转
            elif angle[i] == 'k':
                resultlist.append(cv.flip(imgdata, -1))  # # （任意负数） X-Y翻转
        if num > 3:
            i = 0
            row, col = imgdata.shape[:2]
            for da in random.sample(range(0, int(len(angle2))), num - 3):
                M = cv.getRotationMatrix2D((row / 2, col / 2), int(angle2[da]), 1)  # 生成旋转矩阵，缩放0.9
                imgdst = cv.warpAffine(imgdata, M, (int(row), int(col)), cv.INTER_LINEAR,
                                       borderValue=10)  # cv.INTER_CUBIC, borderValue= 开始旋转变换
                resultlist.append(imgdst)
    return resultlist










"""
#### 分割扁/廋型图片，保留原图（不拆分时不需要resize），并可选择拆分成2个图片（必须resize）。
#### 返回值：包含拆分后的（多张）图片数据的数组array
"""
def img_division(img, size=None, Imgcut=False, ratio=1.7):
    global img1, img2
    Img_array = []
    row = img.shape[0]
    col = img.shape[1]
    if size == None:  # 以防万一
        size = (col, row)
    img1 = cv.resize(img, size)
    Img_array.append(img1)
    if Imgcut == True:
        if row > ratio * col or col > ratio * row:
            if row > ratio * col:
                img1 = img[0: int(row / 2), 0:col, :].copy()
                img2 = img[int(row / 2):row, 0:col, :].copy()
            elif col > ratio * row:
                img1 = img[0:row, 0: int(col / 2), :].copy()
                img2 = img[0:row, int(col / 2): col, :].copy()
            img1 = cv.resize(img1, size)
            img2 = cv.resize(img2, size)
            Img_array.append(img1)
            Img_array.append(img2)
    Img_array = np.array(Img_array)
    return Img_array


"""
#### 清空/删除文件夹——连同文件夹内的文件一并删除
#### 返回值：True成功 ，False失败
"""
def pathdata_delete(path, oriDelete=False):
    try:
        try:
            listName_sub = os.listdir(path)
            if listName_sub != []:  # 空文件夹
                for pathName in listName_sub:  ##逐个读取目录下的子文件/夹
                    path_sub = os.path.join(path, pathName)
                    # print('- ', path_sub)
                    if os.path.isfile(path_sub):
                        os.remove(path_sub)
                        # print('已删除1：', path_sub)
                    else:
                        pathdata_delete(path_sub, True)

        except OSError as o1:
            os.remove(path)
            # print('已删除2：', path)
        if oriDelete:
            os.rmdir(path)
            # print('已删除3：', path)
        print('已删除：', path)

    except Exception as e:
        print(e)
        return False
    # if
    return True



"""
#### 修正图片，将origin地址的所有图片复制到target新地址下： 整齐化文件名
#### 这个过程：可以选择更改图片尺寸resize，修改png->jpg，
#### 以及，对于扁/廋矩形图片进行分割（会保留原图）（1图->3图）
#### 无返回值
"""


def img_changePath(path_origin, path_target=None, resize=None,
                   png2jpg=False, ImgDiv=False):
    global path_imgCopy  # , Imcut
    path_imgCopy = None
    # Imcut = False
    if path_target == None and resize != None:
        print('path_target目录为空，不能变更图片尺寸', '-- img_changePath')
    try:
        listName_sub = os.listdir(path_origin)

        i = 0
        for pathName in listName_sub:  ##逐个读取目录下的子文件/夹
            path_sub = os.path.join(path_origin, pathName)
            # i=i+1
            if ('.jpg' in pathName ) or ('.JPG' in pathName ):  ##如果是图片，则读出
                try:
                    if path_target is not None:
                        img = cv.imread(path_sub)
                        # if ImgDiv == True:
                        #     Imcut = True
                        # else:  # 避免因为其他地方致使Imcut变为True后，一直为True
                        #     Imcut = False

                        imgarray = img_division(img=img, size=resize, Imgcut=ImgDiv)  ##扁矩形resize并分割成2个图像
                        for a in range(imgarray.shape[0]):
                            i = i + 1
                            name = ('%03d' % i) + '.jpg'  # 文件名 自动补0
                            # if findstr(listName_sub, name):  #文件名冲突，变更
                            #     name = name.replace('.jpg', '_2.jpg')
                            img_path = os.path.join(path_target, name)
                            if os.path.isdir(path_target) != True:  # 目录不存在就创建
                                os.makedirs(path_target)
                            cv.imwrite(img_path, imgarray[a])

                            print('写入：', img_path)

                    # print(path_sub)
                except Exception as e2:
                    print('不是jpg图像文件')
            elif ('.png' in pathName ) or ('.PNG' in pathName):  ##如果是图片，则读出
                try:
                    if png2jpg == True or path_target != None:
                        img = cv.imread(path_sub)  # 只有3通道 不知道为什么
                        if path_target != None and ImgDiv == True:
                            Imcut = True
                        else:
                            Imcut = False
                        imgarray = img_division(img=img, size=resize, Imgcut=Imcut)  ##扁矩形resize并分割成2个图像
                        for a in range(imgarray.shape[0]):
                            i = i + 1
                            # if png2jpg == True :  ##都是3通道数据，所以移动照片的话，干脆全部改成jpg格式
                            name = ('%03d' % i) + '.jpg'  # 文件名 自动补0
                            # else:
                            #     name = ('%03d' % i) + '.png'
                            if (name in listName_sub ):  # 文件名冲突，变更
                                name = name.replace('.jpg', '_2.jpg')
                            if path_target == None:
                                img_path = os.path.join(path_origin, name)
                            else:
                                img_path = os.path.join(path_target, name)
                                if os.path.isdir(path_target) != True:  # 目录不存在就创建
                                    os.makedirs(path_target)
                            cv.imwrite(img_path, imgarray[a])
                            print('写入：', img_path)
                except Exception as e2:
                    print('不是png图像文件')
            elif ( '.' in pathName) == False:
                if path_target != None:
                    path_imgCopy = os.path.join(path_target, pathName)
                print('下一级目录:', '(图源)', path_sub, ' --- 复制:', path_imgCopy)  # path_imgCopy
                img_changePath(path_sub, path_imgCopy, resize, png2jpg, ImgDiv=ImgDiv)  ##死亡嵌套

    except Exception as e:
        print('无法打开文件：', path_origin, '-- img_changePath')
    return 0


"""
#### 划分图片，将origin地址的所有图片转移（iscopy=False剪切）（复制iscopy=True)到target新地址下，
#### 分类存放用于train和predict的图片： 使用原文件名。（训练样本数num_train）（测试样本数num_predict）
#### 这个过程：ImgDiv=True对于扁/廋矩形图片进行分割（False会保留原图）（1图->3+图）（文件名加 ’_几‘）
#### 返回值: origin文件夹内所有 【文件夹数，（train样本数，predict样本数，剩余分类图片数量）】
"""


def imgDivide_TrainPredict_list(path_origin, num_train,
                                path_target=None, num_predict=0, iscopy=False,
                                resize=None, ImgDiv=False, path_train=None, path_predict=None):
    global listName_sub, pathName
    num_train_prelist = []
    if num_train == 0:
        print('ERORR >： 训练图片数量不能为 0 ！ ----- imgDivide_TrainPredict')
    if num_predict == 0:
        print('预测图片数量为 0，自动将剩余图片划分为pridict数据')
    if ImgDiv is True and resize is None:
        print('ERROR >: ImgDiv==True 需要拆割图片，必须输入拆割尺寸！ ----- imgDivide_TrainPredict')
    # if path_train is None and path_predict is None:
    #     if path_target is None :
    #         path_train = os.path.join(path_origin,'train')
    #         path_predict = os.path.join(path_origin, 'predict')
    #     else:

    if path_target == None:
        path_train = None
        path_predict = None
    elif path_train is None and path_predict is None:
        path_train = os.path.join(path_target, 'train')
        path_predict = os.path.join(path_target, 'predict')

    try:
        listName_sub = os.listdir(path_origin)
        i = 0
        for pathName in listName_sub:  ##逐个读取目录下的子文件/夹
            path_sub = os.path.join(path_origin, pathName)
            if ('.jpg' in pathName, ) or ('.JPG' in pathName ) or \
                    ('.png' in pathName) or ('.PNG' in pathName ):  ##如果是图片，则读出
                try:

                    if True:
                        img = cv.imread(path_sub)
                        imgarray = img_division(img=img, size=resize, Imgcut=ImgDiv)  ##扁矩形resize并分割成2个图像
                        for a in range(imgarray.shape[0]):
                            i = i + 1
                            if path_target != None:
                                name = pathName  # 使用原文件名
                                if a > 0:  # 图片被分割成多个 改名
                                    name = name.replace('.jpg', str('_%d.jpg' % a))
                                if i <= num_train:
                                    img_path = os.path.join(path_train, name)
                                    if os.path.isdir(path_train) != True:  # 目录不存在就创建
                                        os.makedirs(path_train)
                                else:
                                    img_path = os.path.join(path_predict, name)
                                    if os.path.isdir(path_predict) != True:  # 目录不存在就创建
                                        os.makedirs(path_predict)
                                    if num_predict != 0 and (i - num_train) > num_predict:
                                        break
                                cv.imwrite(img_path, imgarray[a])
                                print(i, '--', img_path)
                                if iscopy is not True:
                                    if os.path.isfile(path_sub):  # 实现剪切功能
                                        os.remove(path_sub)
                                        print('移除：', path_sub)

                    # print(path_sub)
                except Exception as e2:
                    print('不是 jpg/png 图像文件 ', path_sub)
                    print(e2)
            elif  '.' not in pathName:
                print('下一级目录:', '(图源)', path_sub)
                if path_train is not None:
                    path_train0 = os.path.join(path_train, pathName)  ##一定要改名path_train0，避免叠加
                    path_predict0 = os.path.join(path_predict, pathName)  ##一定要改名path_predict0，避免叠加
                    print('(训练集地址:)', path_train0, '(测试集地址:)', path_predict0)
                else:
                    path_train0 = None
                    path_predict0 = None

                num_train_prelist0 = imgDivide_TrainPredict(path_sub, num_train, path_target,
                                                            num_predict=num_predict, iscopy=iscopy,
                                                            resize=resize, ImgDiv=ImgDiv,
                                                            path_train=path_train0, path_predict=path_predict0)  ##死亡嵌套
                num_train_prelist.append(num_train_prelist0)

        if i > 0:
            if i < num_train:
                print('图片 \'训练样本\' 数量不足 ！')
            if (i - num_train) < num_predict:
                print('图片 \'预测样本\' 数量不足 ！')
            if i < num_train:
                # num_train_prelist.append([i, 0, 0])
                num_train_prelist.append(i)
                num_train_prelist.append(0)
                num_train_prelist.append(0)
            elif num_predict != 0 and (i - num_train) > num_predict:
                # num_train_prelist.append([num_train, num_predict, (i - num_train-num_predict)])
                num_train_prelist.append(num_train)
                num_train_prelist.append(num_predict)
                num_train_prelist.append((i - num_train - num_predict))
            else:
                # num_train_prelist.append([num_train, (i - num_train), 0])
                num_train_prelist.append(num_train)
                num_train_prelist.append((i - num_train))
                num_train_prelist.append(0)

        print('已在 ', path_origin, ' 获取样本数量 %d 个 ' % i)
        if iscopy is not True:
            try:
                os.rmdir(path_origin)  ##文件夹要是空了则删除成功
                print('已删除空文件夹:', path_origin)
            except OSError as o1:
                print(o1)
        print()  # 输出空一行，//无意义
    except OSError as e:
        print(e)
        print('无法打开文件：', path_origin, '----- imgDivide_TrainPredict')
    except Exception as e2:
        print(e2)
        print('-无法打开文件：', path_origin, '----- imgDivide_TrainPredict')

    return num_train_prelist


##将上述函数的返回值array化——偷懒，避免麻烦
def imgDivide_TrainPredict(path_origin, num_train,
                           path_target=None, num_predict=0, iscopy=False,
                           resize=None, ImgDiv=False, path_train=None, path_predict=None):
    num_train_prelist = imgDivide_TrainPredict_list(path_origin, num_train, path_target, num_predict, iscopy,
                                                    resize, ImgDiv, path_train, path_predict)
    return np.array(num_train_prelist)




"""
#### 读取origin地址内的全部图片，并且，示旗下每个文件夹内的图片为一类
#### 过程：可以选择再修改图片尺寸resize，设置每类图片样本读取个数（不是总样本数）
#### 返回值： Img_list图片样本（list型），Label_list每类标签（list型），最后的标签值Label（也是类别数）
"""
def getLabel(name: str):
    if name.startswith('CT_COVID'):
        return 1
    elif name.startswith('CT_NonCOVID'):
        return 0
    elif name.startswith('1') or name.startswith('01') or name.startswith('001'):
        return 0
    elif name.startswith('2') or name.startswith('02') or name.startswith('002'):
        return 1
    elif name.startswith('3') or name.startswith('03') or name.startswith('003'):
        return 2
    elif name.startswith('4') or name.startswith('04') or name.startswith('004'):
        return 3
    elif name.startswith('5') or name.startswith('05') or name.startswith('005'):
        return 4
    elif name.startswith('6') or name.startswith('06') or name.startswith('006'):
        return 5
    elif name.startswith('7') or name.startswith('07') or name.startswith('007'):
        return 6
    elif name.startswith('8') or name.startswith('08') or name.startswith('008'):
        return 7
    else:
        return None


def read_ImgDataBase(path_origin, resize=None, class_each=0,
                     Img_list=None, Label_list=None, Label=0, Labelmax = 0):
    if Label_list is None:
        Label_list = []
    if Img_list is None:
        Img_list = []
    if class_each == 0:
        print('class_each为 0，读取文件内全部的图片。每类样本图片数量不能不齐！！', '----read_ImgDataBase')
        # return None, None, 0
    num = 0
    try:
        listName_sub = os.listdir(path_origin)
        for pathName in listName_sub:  ##逐个读取目录下的子文件/夹
            path_sub = os.path.join(path_origin, pathName)
            # if findstr(pathName, '.jpg') or findstr(pathName, '.JPG') or \
            #         findstr(pathName, '.png') or findstr(pathName, '.PNG'):  ##如果是图片，则读出
            if pathName.endswith('.jpg') or pathName.endswith('.JPG') or \
                    pathName.endswith('.png') or pathName.endswith('.PNG'):
                try:
                    img = cv.imread(path_sub)
                    if resize is not None:
                        img = cv.resize(img, resize)
                    Img_list.append(img)
                    Label_list.append([Label])  # 变成数组
                    num = num + 1
                    if num >= class_each and class_each > 0:
                        break
                    # print(path_sub)
                except Exception as e2:
                    print('不是jpg图像文件 :', path_sub)

            elif '.' not in pathName:
                # if num>0:   ##说明当前目录下有图片，所以归为一类
                #     Label =Label+1
                # if num < class_each and class_each > 0:  ##当前目录下有图片也文件夹时，不足数量提示
                #     print('目录样本数量可能不足！ （此级目录文件数为： %d）' % num,
                #           '\n ---------------目录：', path_origin, '----read_ImgDataBase')
                Label = getLabel(pathName)  ##使用文件名做标签，不再根据读取顺序来定义标签
                if Labelmax < Label:  ## 就算文件夹乱序，也没事。只看第一级目录，自动取最大标签(类数)
                    Labelmax = Label
                print('下一级目录:', '(标签:%d [Max:%d])' % (Label,Labelmax), path_sub )
                ##死亡嵌套
                Img_list, Label_list, Label0 = read_ImgDataBase(path_sub, resize,
                                                                class_each=class_each,
                                                                Img_list=Img_list,
                                                                Label_list=Label_list,
                                                                Label=Label,Labelmax=Labelmax)
        # if num > 0:  ##说明当前目录下有图片，所以归为一类
        #     Label = Label + 1
        if num < class_each and class_each > 0 and num>0:
            print('目录样本数量可能不足！ （此级目录文件数为： %d）' % num,
                  '\n ---------------目录：', path_origin, '----read_ImgDataBase')
    except Exception as e:
        print('无法打开文件：', path_origin, '----read_ImgDataBase')
    return Img_list, Label_list, Labelmax + 1


def read_ImgDataZip(path_origin, resize=None, class_each=0):
    Label_list = []
    Img_list = []
    num = 0
    Label = 0
    readzip = zipfile.ZipFile(path_origin, 'r')
    for subpath in readzip.namelist():
        # if a.endswith('\\') or a.endswith('/') or a.endswith('\\\\') or a.endswith('//'):
        #     print('---------')
        # print('==============', reazip.filename)
        if subpath.endswith('jpg'):  ## 是的图片文件
            with readzip.open(subpath, 'r') as imgfile:  ## 则打开
                content = imgfile.read()  ### 读取内容
                img = np.asarray(bytearray(content), dtype='uint8')  ##复制.创建数组
                img = cv.imdecode(img, cv.IMREAD_COLOR)  ##重新编码
                if resize is not None:
                    img = cv.resize(img, resize)
                # cv.imshow('ds', img)
                Img_list.append(img)
                Label_list.append([Label])  # 变成数组
                num = num + 1
        else:
            if num > 0:  ## 说明已经读取到一类图片
                Label = Label + 1
                num = 0  ##下一类图片开始，清0是为了避免图片在多层文件夹内
    return Img_list, Label_list, Label + 1
















