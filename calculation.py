# -*- coding:utf-8 -*-
import numpy as np

class calculation:
    """計算クラス"""

    def LSM_2d(self, x_list, y_list):
        """最小二乗法による２次関数フィッティング
        @ param1[in] x_list     X座標リスト
        @ param2[in] y_list     Y座標リスト
        @ param[out] a, b, b    各係数
        """
        a, b, c = np.polyfit(x_list, y_list, 2)
        return a, b, c

    def calcMidPoint(self, point1, point2):
        """中点算出
        @ param1[in] point1     座標１
        @ param2[in] point2     座標２
        @ param[out] midPoint   中点
        """
        midPoint = []
        x = (point1[0] + point2[0]) / 2
        y = (point1[1] + point2[1]) / 2
        x = int(x)
        y = int(y)
        midPoint.append(x)
        midPoint.append(y)
        return midPoint

    def calcCurvature(self, x_list, y_list, x):
        """曲率算出
        @ param1[in] x_list     X座標リスト
        @ param2[in] y_list     Y座標リスト
        @ param3[in] x          曲率を算出したいX座標
        @ param[out] k          曲率
        """
        a, b, c = self.LSM_2d(x_list, y_list)
        #print("a=", a, "b=", b, "c=", c)
        diff_1 = 2 * a * x + b
        diff_2 = 2 * a
        nume = (1 + pow(diff_1, 2))
        nume = pow(nume, 3/2)
        k = diff_2 / nume
        return k

    def calcFunc1d(self, point1, point2):
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point2[1] - a * point2[0]
        return a, b
