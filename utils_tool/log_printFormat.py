# 程序运行过程打印输出的TXT文件路径（包含文件名）
import csv
import os
import numpy as np


# outprintTxt = "./imagefilelists"

def fprint(outprintTxt, mode, *data0):
    path = os.path.dirname(outprintTxt)  # 获取TXT文件的父级目录
    if os.path.isdir(path) != True:  # 目录不存在就创建
        os.makedirs(path)
    f = open(outprintTxt, mode)  # 以TXT文件末位继续打印内容的方式打开TXT文件
    for data in data0:  # 轮遍所有内容
        if type(data) == type('s'):
            f.write(data)
        else:
            f.write(str(data))
        f.write(' ')  # 以空格间隔每个内容
    f.write('\n')  # 打印完成将帧移到下一行
    f.close()  # 关闭文档


'''将多个数据保存在一个 csv 文件： '''


def train_val_alldata(filenamme, data, header=False):
    headers = ['index', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'spend_time', 'lr']
    with open(filenamme, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, headers, restval='')
        if not header:
            writer.writerow(data)
        else:
            writer.writeheader()


'''将多个数据保存在一个 csv 文件： '''
def train_val_alldata_writeheader(filenamme):
    headers = ['index', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'spend_time', 'lr']
    with open(filenamme, 'a', newline='') as fp:
        writer = csv.DictWriter(fp, headers, restval='')
        writer.writeheader()



def read_csv_file(path):
    global data
    with open(path,'r') as fp:
        # read = csv.reader(fp)
        read = csv.DictReader(fp)

        for da in read:
            # daa = {na: dat for na in name for dat in da.values()}
            daa = [ds for ds in da.values()]
            print(daa)

            data=daa
        # data = dict(data, **daa)
    return data



"""将数据打印成TXT文件：主要用于保存训练和测试过程的 loss和acc，形成数组"""
def plot_txt(txtpath, *data):  # 可以同时传入多个实参
    path = os.path.dirname(txtpath)  # 获取TXT文件的父级目录
    if not os.path.isdir(path):  # 目录不存在就创建
        os.makedirs(path)
    try:
        index = 0
        for da in data:  # 轮遍所有内容，避免传入多个实参时，维度不一致而出错
            lenth = len(da)  # 识别list或array的维度
            if lenth > index:  # 选取维度最大的值
                index = lenth  #
        f = open(txtpath, 'a')  # 打开TXT文件  #以TXT文件末位继续打印内容的方式打开
        for i in range(index):  # 选取维度最大的值，确保将全部的数据打印输出
            for da in data:  # 每个实参按顺序输出一次
                if i < len(da):  # 确保当前维度没有超过实参的总维度
                    if type(da[i]) == type('s'):  # 写入第一列
                        f.write(da[i])  # 打印实参的第i维度
                    else:
                        f.write(str(da[i]))  # 打印实参的第i维度 非str型转str后打印
                f.write('\t')
            f.write('\n')  # 将指针移到下一行
    except EOFError as e:  # 实参是单个数时，lenth = len( da ) 报错，程序进入到此
        f = open(txtpath, 'a')
        for da in data:
            if type(da) == type('s'):  # 字符型
                f.write(da)  # 打印输出
            else:
                f.write(str(da))  # 非sttr型 转str 打印输出
            f.write('\t')  # 每个实参之间 \t 间隔
        f.write('\n')  # 将指针移到下一行
    except TypeError as e2:  # 实参是单个数时，lenth = len( da ) 报错，程序进入到此
        f = open(txtpath, 'a')
        for da in data:
            if type(da) == type('s'):  # 字符型
                f.write(da)
            else:
                f.write(str(da))  # 非sttr型 转str 打印输出
            f.write('\t')  # 每个实参之间 \t 间隔
        f.write('\n')  # 将指针移到下一行
    f.close()
    return 0


"""对工程文件夹内的数据txt文件进行批量处理：修改test的第三列数据"""


def changeProject_dataTXT(path):
    # path = r'D:\Python\00-work\04-train\Pro-41\output'
    if 'output' not in path:
        path = os.path.join(path, 'output')
    subFolderlist = os.listdir(path)
    for subfoldename in subFolderlist:
        # if subfoldename != 'mynet2' :
        #     continue
        print('工作文件夹：', subfoldename)
        subfolderpath = os.path.join(path, subfoldename)
        subfilelist = os.listdir(subfolderpath)
        print('文件夹内的文件：', subfilelist)
        for subfilename in subfilelist:
            if 'test' in subfilename and '2.txt' not in subfilename:
                filepath = os.path.join(subfolderpath, subfilename)
                print('操作路径：', filepath)
                data_arr = np.loadtxt(filepath)  # ,dtype=float
                print('文件数据维度(shape)：', data_arr.shape)
                if data_arr[0][2] == 0:
                    continue
                f = open(filepath, 'w')  # 以TXT文件末位继续打印内容的方式打开TXT文件
                for i in range(data_arr.shape[0]):
                    data_arr[i, 2] = float(data_arr[i, 2] - 1)
                    # print(data_arr)
                    f.write(str(data_arr[i, 0]))
                    f.write('\t')
                    f.write(str(data_arr[i, 1]))
                    f.write('\t')
                    f.write(str(data_arr[i, 2]))
                    f.write('\n')
                f.close()
                try:
                    os.remove(filepath + '2.txt')
                except:
                    pass
                    # f.write('{:.16f}\t{:.16f}\t{:1.0f}\n'.format(data_arr[i,0],data_arr[i,1],data_arr[i,2]))

                # print('保存数据',filepath+'2','\n',data_arr)
                # np.savetxt(filepath +'2', data_arr)
