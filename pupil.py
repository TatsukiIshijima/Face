# -*- coding:utf-8 -*-
import numpy as np
import cv2

class pupil:
    """瞳孔検出クラス"""

    def detectPupil(self, iris_points):
        """
        @ param1[in]  iris_points(np.array)    虹彩座標リスト
        @ param1[out] cx                       重心X座標
        @ param2[out] cy                       重心Y座標
        @ param3[out] iris_hull                虹彩の凸包座標リスト
        """
        # 凸包座標算出
        """iris_pointsの個数が少ない場合エラー"""
        iris_hull = cv2.convexHull(iris_points)
        # 重心算出
        """分母が0の場合エラー"""
        M = cv2.moments(iris_hull)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy, iris_hull
