# -*- coding:utf-8 -*-
import numpy as np

class iris:
    """虹彩検出クラス"""

    """目領域内の虹彩検出の閾値決定
    @ param1[in] ROI        目の領域のROI(グレースケール)
    @ param2[in] mask       上記領域のマスク
    @ param3[in] rate       輝度比
    @ param[out] threshold  閾値
    """
    def defineThreshold(self, ROI, mask, rate):
        eye_luminance = []
        height, width = ROI.shape[:2]

        for row in range(height):
            for col in range(width):
                if mask[row][col] == 255:
                    eye_luminance.append(ROI[row][col])

        threshold = (max(eye_luminance) - min(eye_luminance)) * rate
        return threshold

    """虹彩検出
    @ param1[in] ROI_left               原画像中の抽出したROIの左上頂点のX座標
    @ param2[in] ROI_top                原画像中の抽出したROIの左上頂点のY座標
    @ param3[in] ROI                    目の領域のROI(グレースケール)
    @ param4[in] mask                   上記領域のマスク
    @ param5[in] threshold              閾値
    @ param[out] iris_points(np.array)  虹彩座標リスト
    """
    def detectIris(self, ROI_left, ROI_top, ROI, mask, threshold):
        iris_points = []
        height, width = ROI.shape[:2]

        for row in range(height):
            for col in range(width):
                if mask[row][col] == 255:
                    if ROI[row][col] <= int(threshold):
                        iris_point = []
                        iris_point.append(ROI_left + col)
                        iris_point.append(ROI_top + row)
                        iris_point = np.array(iris_point)
                        iris_points.append(iris_point)

        iris_points = np.array(iris_points)
        return iris_points
