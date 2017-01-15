# -*- coding:utf-8 -*-
import numpy as np
import cv2
import sys
import pylab as plt
import math

class otsu:

    def threshOtsu(self, image_1d):
        """大津の二値化
        @ param1[in]  image_1d   画像の1次元配列
        @ param1[out] thresh     閾値
        参考：http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
        """
        fn_min = np.inf
        thresh = -1

        hist = plt.hist(image_1d, 256, [0, 256])
        Q = hist[0].cumsum()                      # 累積度数配列(画素数の和)
        bins = np.arange(256)

        for i in range(0, 256):
            p1, p2 = np.hsplit(hist[0], [i])      # 輝度度数
            q1, q2 = Q[i], Q[255] - Q[i]          # 画素数
            b1, b2 = np.hsplit(bins, [i])         # 輝度値

            if q1 != 0.0:
                m1 = np.sum(p1 * b1) / q1
                v1 = np.sum(((b1 - m1)**2) * p1) / q1
            else:
                m1 = 0.0
                v1 = 0.0
            if q2 != 0.0:
                m2 = np.sum(p2 * b2) / q2
                v2 = np.sum(((b2 - m2)**2) * p2) / q2
            else:
                m2 = 0.0
                q2 = 0.0
            fn = v1 * q1 + v2 * q2

            if fn < fn_min:
                fn_min = fn
                thresh = i

        return thresh

if __name__ == '__main__':

    otsu = otsu()

    param = sys.argv
    image = cv2.imread(param[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    copy = image.copy()
    ret, thre = cv2.threshold(copy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    image_1d = image.ravel()                            # ２次元から１次元に
    thresh = otsu.threshOtsu(image_1d=image_1d)
    print("OpenCV : ", ret, "Otsu : ", thresh)

    fig = plt.figure(figsize=(14, 7))
    ax_img = fig.add_subplot(121)
    ax_hist = fig.add_subplot(122)
    ax_img.set_title("Image")
    ax_hist.set_title("Histgram")

    ax_hist.set_xlabel("Luminance Value")
    ax_hist.set_ylabel("Number of Pixels")

    ax_img.imshow(image, cmap=plt.cm.gray)

    ax_hist.set_xlim([0, 256])
    ax_hist.hist(image_1d, 256, [0, 256])

    plt.show()
