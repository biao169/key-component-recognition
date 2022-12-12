import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



""" 竖屏版 way='h'  横屏版 way='w' """
## 可通过按键移动裁剪框，框为正方形
def videoCaptureCut(aviName, savePath, imgName, way='h'):
    camera = cv.VideoCapture(aviName)  # 从文件读取视频
    if (camera.isOpened()):
        print('isOpen')
    else:
        print('Fail to open!')

    # 测试用,查看视频size
    size = (int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print('size:' + repr(size))

    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    step = 30;
    w = 10;
    k = 0;
    row = 0;
    col = 0
    if way == 'w':
        col = 110
    else:
        row = 110
    count = 1
    interval = 5  # 时间间隔，减少帧率
    while True:
        grabbed, frame_lwpCV = camera.read()  # 逐帧采集视频流
        if not grabbed:
            break
        # print(grabbed,frame_lwpCV.shape)
        # img_lwpCV = cv.cvtColor(frame_lwpCV, cv.COLOR_BGR2RGB) # 转灰度图
        interval = interval - 1
        if interval > 0:
            continue
        else:
            interval = 5
        rowmax, colmax = frame_lwpCV.shape[:2]
        if count == 1:
            print('图片尺寸：', rowmax, colmax)
        # M = cv.getRotationMatrix2D( ( int(colmax/2),int( rowmax/2)  ), 270,0.5)
        # frame_lwpCV = cv.warpAffine(frame_lwpCV,M,( int(colmax),int( rowmax) ) )
        frame_lwpCV = cv.resize(frame_lwpCV, (int(colmax / 2), int(rowmax / 2)))
        img_lwpCV = frame_lwpCV.copy()
        rowmax, colmax = img_lwpCV.shape[:2]
        timeCount = 2
        while (timeCount > 0):
            key = cv.waitKeyEx(50)
            if key != -1:
                timeCount = 3
                if key & 0xff == ord('a'):
                    k = (k - 1) if k > 0 else 0
                elif key & 0xff == ord('d'):
                    k = (k + 1) if k < 2 else 2
                elif key & 0xff == ord('s'):
                    step = (step - 1) if (step - 1) > 0 else 1
                elif key & 0xff == ord('w'):
                    step = (step + 1) if (step + 1) < 50 else 50
                elif key & 0xff == ord('n'):
                    w = (w - 1) if (w - 1) > 0 else 1
                elif key & 0xff == ord('m'):
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
                print('step:', step, ' row:', row)
                if way == 'w':
                    if col + rowmax >= colmax:
                        col = colmax - rowmax
                else:
                    if row + colmax >= rowmax:
                        row = rowmax - colmax
            else:
                timeCount = timeCount - 1
            img_lwpCV2 = img_lwpCV.copy()
            if way == 'w':
                lwpCV_cut = cv.rectangle(img_lwpCV2, (col, 0), (col + rowmax, rowmax), (0, 255, 0), 2)
            else:
                lwpCV_cut = cv.rectangle(img_lwpCV2, (0, row), (colmax, row + colmax), (0, 255, 0), 2)
            cv.imshow('cut', lwpCV_cut)
        if way == 'w':
            imgCut = img_lwpCV[0:rowmax - 1, col:col + rowmax - 1]
        else:
            imgCut = img_lwpCV[row:row + colmax - 1, 0:colmax - 1]
        cv.imshow('Cut_result', imgCut)
        imgCut = cv.resize(imgCut, (256, 256))
        name = ('%03d' % count) + '.jpg'  # 文件名 自动补0
        cv.imwrite((savePath + '/' + imgName + name), imgCut)
        count = count + 1
        # img = cv.imread((savePath + '/' + imgName + name))
        # cv.imshow('fgsdf',img)
        # key = cv.waitKey(10) & 0xFF
        # if key == ord('q'):
        #     break
    camera.release()
    cv.destroyAllWindows()


### 将视频缩小尺寸，形状不变；
def videoCapture_turnImg(aviPath, savePath, imgName, startNum=0, imgsize=256):
    camera = cv.VideoCapture(aviPath)  # 从文件读取视频
    if (camera.isOpened()):
        print('isOpen')
    else:
        print('Fail to open!')
    if not os.path.isdir(savePath):
        os.makedirs(savePath)

    # 测试用,查看视频size
    size = (int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print('size:' + repr(size))

    k = min(size)/imgsize

    count = startNum
    jump = 0
    while True:
        grabbed, frame_lwpCV = camera.read()  # 逐帧采集视频流
        if not grabbed:
            break
        jump += 1
        if jump < 16:
            continue
        jump = 0
        rowmax, colmax = frame_lwpCV.shape[:2]
        frame_lwpCV = cv.resize(frame_lwpCV, (int(colmax / k), int(rowmax / k)))
        count += 1
        print('\rfinish: %d' %count, end='')
        name = ('%03d' % count) + '.jpg'  # 文件名 自动补0
        cv.imwrite((savePath + '/' + imgName + name), frame_lwpCV)
    print('\norign:', aviPath, '\tsave in path:', savePath)


if __name__ == '__main__':
    avi101 = r'E:\01_workPicture\avi\01\01.mp4'
    avi102 = r'E:\01_workPicture\avi\01\02.mp4'
    avi103 = r'E:\01_workPicture\avi\01\03.mp4'
    avi104 = r'E:\01_workPicture\avi\01\04.mp4'
    avi105 = r'E:\01_workPicture\avi\01\05.mp4'
    avi106 = r'E:\01_workPicture\avi\01\06.mp4'

    avi01 = r'E:\01_workPicture\avi\02\01.mp4'
    avi012 = r'E:\01_workPicture\avi\02\01b.mp4'
    avi02 = r'E:\01_workPicture\avi\02\02.mp4'
    avi03 = r'E:\01_workPicture\avi\02\03.mp4'
    avi04 = r'E:\01_workPicture\avi\02\04.mp4'
    avi05 = r'E:\01_workPicture\avi\02\05.mp4'
    avi06 = r'E:\01_workPicture\avi\02\06.mp4'
    imgsavePath = 'E:/01_workPicture/dataset_5_rule'

    # videoCaptureCut(avi01,imgsavePath+'/'+'01','a',way='w')
    # videoCaptureCut(avi012, imgsavePath + '/' + '01', 'a1', way='w')
    # videoCaptureCut(avi02, imgsavePath + '/' + '02', 'b', way='w')
    # videoCaptureCut(avi03, imgsavePath + '/' + '03', 'c', way='w')
    # videoCaptureCut(avi04, imgsavePath + '/' + '04', 'd', way='w')
    # videoCaptureCut(avi05, imgsavePath + '/' + '05', 'e', way='w')
    # videoCaptureCut(avi06, imgsavePath + '/' + '06', 'f', way='w')

    # videoCapture_turnImg(avi01, imgsavePath+'/'+'01', 'a', startNum=0)
    # videoCapture_turnImg(avi012, imgsavePath + '/' + '01', 'a', startNum=263)
    # videoCapture_turnImg(avi02, imgsavePath + '/' + '02', 'b')
    # videoCapture_turnImg(avi03, imgsavePath + '/' + '03', 'c')
    # videoCapture_turnImg(avi04, imgsavePath + '/' + '04', 'd')
    # videoCapture_turnImg(avi05, imgsavePath + '/' + '05', 'e')
    # videoCapture_turnImg(avi06, imgsavePath + '/' + '06', 'f')

    videoCapture_turnImg(avi101, imgsavePath + '/' + '01', 'a', startNum=347)
    videoCapture_turnImg(avi102, imgsavePath + '/' + '02', 'b', startNum=409)
    videoCapture_turnImg(avi103, imgsavePath + '/' + '03', 'c', startNum=373)
    videoCapture_turnImg(avi104, imgsavePath + '/' + '04', 'd', startNum=377)
    videoCapture_turnImg(avi105, imgsavePath + '/' + '05', 'e', startNum=450)
    videoCapture_turnImg(avi106, imgsavePath + '/' + '06', 'f', startNum=417)
