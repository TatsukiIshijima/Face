# -*- coding:utf-8 -*-
import numpy as np

class iris:
    """虹彩検出クラス"""

    """目領域内の虹彩検出の閾値決定
    @ param1[in] ROI            目の領域のROI(グレースケール)
    @ param2[in] mask           上記領域のマスク
    @ param3[in] rate           輝度比
    @ param1[out] threshold     閾値
    @ param2[out] eye_luminance マスク内輝度値リスト
    """
    def defineThreshold(self, ROI, mask, rate):
        eye_luminance = []
        height, width = ROI.shape[:2]

        for row in range(height):
            for col in range(width):
                if mask[row][col] == 255:
                    eye_luminance.append(ROI[row][col])

        eye_luminance = np.array(eye_luminance)
        threshold = (max(eye_luminance) - min(eye_luminance)) * rate
        return threshold, eye_luminance

    """虹彩検出
    @ param1[in] ROI_left               原画像中の抽出したROIの左上頂点のX座標
    @ param2[in] ROI_top                原画像中の抽出したROIの左上頂点のY座標
    @ param3[in] ROI                    目の領域のROI(グレースケール)
    @ param4[in] mask                   上記領域のマスク
    @ param5[in] threshold              閾値
    @ param6[in] abso                   画像における絶対座標で返すか返さないか
    @ param[out] iris_points(np.array)  虹彩座標リスト
    """
    def detectIris(self, ROI_left, ROI_top, ROI, mask, threshold, abso):
        iris_points = []
        height, width = ROI.shape[:2]

        for row in range(height):
            for col in range(width):
                if mask[row][col] == 255:
                    if ROI[row][col] <= int(threshold):
                        iris_point = []
                        if abso == True:
                            iris_point.append(ROI_left + col)
                            iris_point.append(ROI_top + row)
                        else:
                            iris_point.append(col)
                            iris_point.append(row)
                        iris_point = np.array(iris_point)
                        iris_points.append(iris_point)

        iris_points = np.array(iris_points)
        return iris_points
