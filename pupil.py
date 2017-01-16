# -*- coding:utf-8 -*-
import numpy as np
import cv2
import extract
import iris

extract = extract.extract()
iris = iris.iris()

class pupil:
    """瞳孔検出クラス"""

    def detectPupil(self, iris_points):
        """重心による瞳孔検出
        @ param1[in]  iris_points(np.array)    虹彩座標リスト
        @ param1[out] cx                       重心X座標
        @ param2[out] cy                       重心Y座標
        @ param3[out] iris_hull                虹彩の凸包座標リスト
        """
        # 凸包座標算出
        iris_hull = cv2.convexHull(iris_points)
        # 重心算出
        M = cv2.moments(iris_hull)
        if (M['m00'] != 0.0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx = None
            cy = None
        return cx, cy, iris_hull

    def getPupilPoint(self, xpoint, ypoint, gray_img, eye_mask, rate):
        """瞳孔座標取得
        @ param1[in]  xpoint                目の輪郭のX座標リスト
        @ param2[in]  ypoint                目の輪郭のY座標リスト
        @ param3[in]  gray_img              入力画像のグレースケール画像
        @ param4[in]  eye_mask              目の領域のマスク
        @ param5[in]  rate                  輝度比
        @ param1[out] eye_roi               目領域のROI
        @ param2[out] eye_luminance         輝度値リスト
        @ param3[out] iris_mask             虹彩マスク
        @ param4[out] cx                    重心X座標(近似瞳孔x座標)
        @ param5[out] cy                    重心Y座標(近似瞳孔y座標)
        @ param6[out] iris_hull             虹彩の凸包座標
        """
        # 目の領域抽出
        top, bottom, left, right = extract.cutArea(xpoint, ypoint)
        eye_roi = gray_img[top:bottom, left:right]
        eye_mask_roi = eye_mask[top:bottom, left:right]
        iris_mask = np.zeros((eye_mask_roi.shape[0], eye_mask_roi.shape[1], 1), dtype=np.uint8)
        # 目の虹彩、瞳孔検出
        # 閾値設定(割合ver)
        threshold, eye_luminance = iris.defineThreshold(eye_roi, eye_mask_roi, rate)
        print("threshold : ", threshold)
        iris_points = iris.detectIris(left, top, eye_roi, eye_mask_roi, threshold, abso=True)
        iris_points_relative = iris.detectIris(left, top, eye_roi, eye_mask_roi, threshold, abso=False)
        if len(iris_points) == 0:
            cx = None
            cy = None
            iris_hull = []
        else:
            # 虹彩マスク作成
            for i in range(len(iris_points_relative)):
                iris_mask[iris_points_relative[i][1]][iris_points_relative[i][0]] = 255
            # 重心座標と凸包座標
            cx, cy, iris_hull = self.detectPupil(iris_points)
        return eye_roi, eye_luminance, iris_mask, cx, cy, iris_hull
