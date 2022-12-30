import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

"""使用特征检测，获取图像特征区域并切割出来"""


### 特征使用ORB算法，先滤波再计算特征点。特征区域可以放大成图片尺寸的 1/K 倍
### k==1:特征区域不放大； k==0:特征区域默认比图片整体尺寸小一半
def imageConcentration(img00, k: float = 0, ishow=False):
    try:
        img0 = img00.copy()
    except:
        img0 = img00.clone()
    img0 = cv.bilateralFilter(img0, 5, 12.5, 6)  ## 双边滤波
    img0 = cv.GaussianBlur(img0, (5, 5), 10)  # 高斯滤波
    # img0 = cv.blur(img0, (3, 3), (-1, -1))  ## #中值滤波

    surf = cv.ORB_create()
    tkp, tdes = surf.detectAndCompute(img0, None)
    imgdetectRes = cv.drawKeypoints(img0, tkp, None, (255, 0, 0), 2)
    # cv.imshow('detect',imgdetectRes)
    keyPoint = []
    for kp in tkp:
        keyPoint.append(kp.pt)
    keyPoint = np.array(keyPoint)
    try:
        maxpoint = np.max(keyPoint, axis=0)
        minpoint = np.min(keyPoint, axis=0)
        meanpoint = np.array([(maxpoint[0] + minpoint[0]) / 2, (maxpoint[1] + minpoint[1]) / 2])
        subSize = maxpoint - minpoint
        # if k!=1:
        #     if subSize[0]==0:subSize[0]=img00.shape[1]
        #     if subSize[1] == 0: subSize[1] = img00.shape[0]
        #     rate = min( img00.shape[1]  / subSize[0], img00.shape[0] / subSize[1] )
        #     if rate>k:
        #         if k==0:
        #             k = rate/2  #  min(img00.shape[:2])/max(subSize)
        #         else: k = rate/k
        #     else: k=1.3
        subSize[0] = max(subSize);
        subSize[1] = max(subSize)
        minpoint = np.array(
            [math.trunc(meanpoint[0] - k * subSize[0] / 2), math.trunc(meanpoint[1] - k * subSize[1] / 2)])
        maxpoint = np.array(
            [math.trunc(meanpoint[0] + k * subSize[0] / 2), math.trunc(meanpoint[1] + k * subSize[1] / 2)])
    except:
        minpoint = np.array([0, 0])
        maxpoint = np.array([img00.shape[1] - 1, img00.shape[0] - 1])
    if ishow:print('minpoint,maxpoint:', minpoint, maxpoint)
    if minpoint[0] < 0:  minpoint[0] = 0
    if minpoint[1] < 0:  minpoint[1] = 0
    if maxpoint[0] >= img00.shape[1]:  maxpoint[0] = img00.shape[1] - 1
    if maxpoint[1] >= img00.shape[0]:  maxpoint[1] = img00.shape[0] - 1

    if ishow:print('img.shape:', img00.shape)
    if ishow:print('差值：', maxpoint - minpoint)
    imgCut = img00[minpoint[1]:maxpoint[1], minpoint[0]:maxpoint[0]]
    # cv.imshow('imgCut',imgCut)
    imgRect = cv.rectangle(imgdetectRes, (minpoint[0], minpoint[1]), (maxpoint[0], maxpoint[1]), (0, 0, 255), 2)
    if ishow:
        cv.imshow('imgRect', imgRect)
        cv.waitKey(10)
    return imgCut


"""调试函数：键盘控制阀值操作，轮廓检测，"""


def imgCut(img0):
    cv.imshow('orign', img0)

    # img4 = cv.Canny(img0, 30, 80)  ### ### ,apertureSize=None,L2gradient=False
    # img5 = cv.Canny(img0, 30, 80)  ### ### ,apertureSize=None,L2gradient=False
    # cv.imshow('dfs0', img4)
    # cv.imshow('dfs',img5)

    gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    gray = filter(gray)
    ret = 10
    while True:
        key = cv.waitKeyEx(50)
        if key != -1:
            if key == 2490368:  ##上
                ret = (ret + 2) if (ret + 2) <= 255 else 255
            elif key == 2621440:  # ##下
                ret = (ret - 2) if (ret - 2) >= 0 else 0
            print('ret=', ret)
            ret, imgthre = cv.threshold(gray, ret, 255, cv.THRESH_BINARY_INV)  # |cv.THRESH_OTSU
            cv.imshow('threshold00', imgthre)
            kernel = np.ones((5, 5), np.uint8)  ### 腐蚀的核定义
            imgthre = cv.morphologyEx(imgthre, cv.MORPH_OPEN, kernel, iterations=3)  ### 腐蚀
            cv.imshow('morphologyEx', imgthre)
            dist_tranform = cv.distanceTransform(imgthre, cv.DIST_L2, 5)
            cv.imshow('dist_tranform', dist_tranform)
            ret0, imgthre = cv.threshold(dist_tranform, 0.7 * dist_tranform.max(), 255, 0)  # |cv.THRESH_OTSU
            cv.imshow('distanceTransform', imgthre)
            # coutours,hier = cv.findContours(imgthre,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
            # imgf2 = cv.drawContours(img0.copy(),coutours,-1,(0,255,0),1)
            # cv.imshow('coutour',imgf2)

    pass


"""功能体验：特征检测-匹配"""


def featureDetect(fa_detected, sub_dectect, isshow: bool = False):
    # surf = cv.xfeatures2d.SURF_create(3000)
    # surf.setUpright(True)
    # surf.setHessianThreshold(400)
    sub_dectect = filter(sub_dectect)
    fa_detected = filter(fa_detected)
    surf = cv.ORB_create()
    tkp, tdes = surf.detectAndCompute(sub_dectect, None)
    fkp, fdes = surf.detectAndCompute(fa_detected, None)

    # ####  K-临近法 匹配 -------------------------
    bf = cv.BFMatcher(cv.NORM_L1)  # , crossCheck = True
    matches2 = bf.knnMatch(tdes, fdes, k=4)  # bf.match(tdes ,fdes)  #
    matchesMask = [];
    pos = []
    ###  去除错误匹配
    for i, (m, n, n2, n3) in enumerate(matches2):
        print(m.distance, n.distance, n2.distance, n3.distance)
        if m.distance < 0.75 * n.distance and \
                m.distance < 0.75 * n2.distance and \
                m.distance < 0.75 * n3.distance:  # # True and :

            matchesMask.append([m])
            # print(m.distance,'   ',m.queryIdx,'   ',m.trainIdx)
            # print( fkp[m.trainIdx].pt )
            pos.append(fkp[m.trainIdx].pt)
    pos = np.array(pos)
    print(pos.shape)
    x = int(np.min(pos[:, 0]));
    xx = int(np.max(pos[:, 0]))
    y = int(np.min(pos[:, 1]));
    yy = int(np.max(pos[:, 1]))
    img_cut = fa_detected.copy()

    if isshow:
        # cv.rectangle(img_cut,(x,y),(xx,yy),[255,0,0],2)
        # img_cut0 = cv.resize(img_cut,(800,800))
        # cv.imshow('dsds',img_cut0)

        timg_detectRes = cv.drawKeypoints(sub_dectect, tkp, None, (255, 0, 0), 2)
        fimg_detectRes = cv.drawKeypoints(fa_detected, fkp, None, (255, 0, 0), 2)
        # ####  Knn-临近法 匹配 -------------------------
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.knnMatch(tdes, fdes, k=1)  # bf.match(tdes ,fdes)  #
        img_MatcherRes = cv.drawMatchesKnn(sub_dectect, tkp, fa_detected, fkp, matches[:20], None, flags=2)
        # # 将图像显示
        # matchColor是两图的匹配连接线，连接线与matchesMask相关
        # singlePointColor是勾画关键点
        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matches2,
                          flags=0)
        # img3 = cv.drawMatchesKnn(timg,tkp,fimg,fkp,matchesMask,None,**drawParams)
        img3 = cv.drawMatchesKnn(sub_dectect, tkp, fa_detected, fkp, matchesMask, None, flags=2)

        cv.namedWindow('ds', cv.WINDOW_NORMAL)
        # cv.resizeWindow("ds", 1200, 800)  #调整窗口大小
        cv.imshow('ds', img3)
        # for i in range(len(matches)):
        #     img_MatcherRes0 = cv.drawMatchesKnn(timg, tkp, fimg, fkp, matches[(int(i)-2):int(i)], None, flags=2)
        #     for dat in matches[i]:
        #         print(dat.distance)
        #     cv.imshow('ds',img_MatcherRes0)
        #     cv.waitKey()
        # cv.destroyAllWindows()

        row = 8;
        col = 2
        fig = plt.figure(1, (10, 12))
        fig.tight_layout()  # 调整整体空白
        fig.add_subplot(row, col, 1), plt.subplots_adjust(wspace=0.1, hspace=0)  # 调整子图间距
        plt.imshow(sub_dectect), plt.title('orign_target'), plt.axis('off')
        fig.add_subplot(row, col, 2), plt.subplots_adjust(wspace=0.1, hspace=0.22)  # 调整子图间距
        plt.imshow(timg_detectRes), plt.title('detectRes_target'), plt.axis('off')
        row = 5
        fig.add_subplot(row, col, 3), plt.subplots_adjust(wspace=0.1, hspace=0.22)  # 调整子图间距
        plt.imshow(fa_detected), plt.title('orign_father'), plt.axis('off')
        fig.add_subplot(row, col, 4), plt.subplots_adjust(wspace=0.1, hspace=0.22)  # 调整子图间距
        plt.imshow(fimg_detectRes), plt.title('detectRes_father'), plt.axis('off')

        fig.add_subplot(2, 2, 3), plt.subplots_adjust(wspace=0.1, hspace=0.22)  # 调整子图间距
        plt.imshow(img_MatcherRes), plt.title('img_MatcherRes'), plt.axis('off')
        fig.add_subplot(2, 2, 4), plt.subplots_adjust(wspace=0.1, hspace=0.22)  # 调整子图间距
        plt.imshow(img3), plt.title('knn img3'), plt.axis('off')
        plt.draw(), plt.pause(1e-10)
    pass


""" 调试函数：颜色空间匹配，可用键盘控制像素采集点 """


def colorMatch(fimg, timg):
    cv.imshow('f img', fimg)
    cv.imshow('t img', timg)
    timg_lab = cv.cvtColor(timg, cv.COLOR_BGR2HSV)
    fimg_lab = cv.cvtColor(fimg, cv.COLOR_BGR2HSV)

    for i in range(3):
        cv.imshow('{} t  chan'.format(i), timg_lab[:, :, i])
        cv.imshow('{} f  chan'.format(i), fimg_lab[:, :, i])
    # fig=plt.figure('Match',figsize=(10,8))
    # fig.tight_layout()  # 调整整体空白
    # for i in range(3):   ## 竖着排列
    #     fig.add_subplot(3, 5, i*5 + 1), plt.subplots_adjust(wspace=0.1, hspace=0.2)
    #     plt.imshow(timg_lab[:,:,i]),plt.title('{} t  chan'.format(i)), plt.axis('off')
    #     fig.add_subplot(3, 5, i*5 + 2), plt.subplots_adjust(wspace=0.1, hspace=0.2)
    #     plt.imshow(fimg_lab[:, :, i]), plt.title('{} f  chan'.format(i)), plt.axis('off')
    # plt.draw(), plt.pause(1e-10)
    # cv.imshow('fimg res', fimg_lab)
    if True:
        step = 30;
        w = 10;
        k = 0;
        row = 0;
        col = 0
    while True:
        rowmax, colmax = timg_lab.shape[:2]
        timeCount = 2
        while (timeCount > 0):
            key = cv.waitKeyEx(50)
            if key != -1:
                timeCount = 1
                if key & 0xff == ord('a'):  ## 图像通道
                    k = (k - 1) if k > 0 else 0
                elif key & 0xff == ord('d'):  ## 图像通道
                    k = (k + 1) if k < 2 else 2
                elif key & 0xff == ord('s'):  ## 步距
                    step = (step - 1) if (step - 1) > 0 else 1
                elif key & 0xff == ord('w'):  ## 步距
                    step = (step + 1) if (step + 1) < 50 else 50
                elif key & 0xff == ord('n'):  ### 匹配值裕度范围
                    w = (w - 1) if (w - 1) > 0 else 1
                elif key & 0xff == ord('m'):  ### 匹配值裕度范围
                    w = (w + 1) if (w + 1) < 50 else 50
                else:
                    if key == 2490368:  ##上
                        row = (row - step) if (row - step) >= 0 else 0
                    elif key == 2621440:  # ##下
                        row = (row + step) if (row + step) < rowmax else rowmax - 1
                    elif key == 2424832:  ##左
                        col = (col - step) if (col - step) >= 0 else 0
                    elif key == 2555904:  ##右
                        col = (col + step) if (col + step) < colmax else colmax - 1
                print('step:', step, 'k:', k, ' row:', row, 'col:', col, 'width', w)
                if key == 2490368 or key == 2621440 or key == 2424832 or key == 2555904:
                    va1ue = timg_lab[row, col, k]
                    tar = timg_lab.copy()
                    tar[row - 2:row, col - 2:col, k] = 255
                    cv.imshow('{} t  chan'.format(k), tar[:, :, k])
                    # fig = plt.figure('Match', figsize=(10, 8))
                    # fig.tight_layout()  # 调整整体空白
                    # for i in range(3):  ## 竖着排列
                    #     if i!=k:
                    #         fig.add_subplot(3, 5, i * 5 + 1), plt.subplots_adjust(wspace=0.1, hspace=0.22)
                    #         plt.imshow(timg_lab[:, :, i]), plt.title('{} t  chan'.format(i)), plt.axis('off')
                    #     else:
                    #         fig.add_subplot(3, 5, k * 5 + 1), plt.subplots_adjust(wspace=0.1, hspace=0.22)
                    #         plt.imshow(tar[:, :, k]), plt.title('{} t  chan'.format(k)), plt.axis('off')
                    #     fig.add_subplot(3, 5, i * 5 + 2), plt.subplots_adjust(wspace=0.1, hspace=0.22)
                    #     plt.imshow(fimg_lab[:, :, i]), plt.title('{} f  chan'.format(i)), plt.axis('off')
                    # plt.draw(), plt.pause(1e-10)

                    frow, fcol = fimg_lab.shape[:2]
                    res = fimg_lab.copy()
                    res = cv.cvtColor(res, cv.COLOR_HSV2BGR)
                    for ii in range(frow):
                        for jj in range(fcol):
                            if fimg_lab[ii, jj, k] >= va1ue - w and fimg_lab[ii, jj, k] <= va1ue + w:
                                res[ii, jj] = [0, 255, 0]
                    cv.imshow('fimg res', res)
                    # fig.add_subplot(2, 2, 2), plt.subplots_adjust(wspace=0.1, hspace=0.22)
                    # plt.imshow(tar ), plt.title('tar'.format(k)), plt.axis('off')
                    # fig.add_subplot(2, 2, 4), plt.subplots_adjust(wspace=0.1, hspace=0.22)
                    # plt.imshow(res), plt.title('match res'), plt.axis('off')
                    # plt.draw(), plt.pause(1e-10)
                    # plt.show()
    pass


"""特征检测，并绘制特征点"""


def imgThreholdCut(dataset):
    for i in range(len(dataset)):
        imgfilePath = dataset[i][0]
        # print('image file path:',imgfilePath)
        img00 = cv.imread(imgfilePath)
        img00 = filter(img00)
        # img0 = cv.cvtColor(img00,cv.COLOR_BGR2GRAY)
        # img = cv.equalizeHist(img0)
        # cv.imshow('img0', img0)
        # retValue,imgThre= cv.threshold(img0,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
        # cv.imshow('imgThre0', imgThre)
        # retValue0, imgThre = cv.threshold(img0, 80, 255, cv.THRESH_BINARY )
        # print('retValue:',retValue)
        # cv.imshow('imgThre',imgThre)

        surf = cv.ORB_create()
        tkp, tdes = surf.detectAndCompute(img00, None)
        detectRes = cv.drawKeypoints(img00, tkp, None, (255, 0, 0), 2)
        cv.imshow('detect', detectRes)
        cv.waitKey(1000)
