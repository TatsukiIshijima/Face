# -*- coding:utf-8 -*-
import numpy as np
import extract
import otsu

extract = extract.extract()
otsu = otsu.otsu()

class iris:
    """虹彩検出クラス"""

    def defineThreshold(self, ROI, mask, rate):
        """目領域内の虹彩検出の閾値決定
        @ param1[in] ROI            目の領域のROI(グレースケール)
        @ param2[in] mask           上記領域のマスク
        @ param3[in] rate           輝度比
        @ param1[out] threshold     閾値
        @ param2[out] eye_luminance マスク内輝度値リスト
        """
        eye_luminance = []
        height, width = ROI.shape[:2]

        for row in range(height):
            for col in range(width):
                if mask[row][col] == 255:
                    eye_luminance.append(ROI[row][col])

        eye_luminance = np.array(eye_luminance)
        threshold = (max(eye_luminance) - min(eye_luminance)) * rate
        return threshold, eye_luminance

    def detectIris(self, ROI_left, ROI_top, ROI, mask, threshold, abso):
        """虹彩検出
        @ param1[in] ROI_left               原画像中の抽出したROIの左上頂点のX座標
        @ param2[in] ROI_top                原画像中の抽出したROIの左上頂点のY座標
        @ param3[in] ROI                    目の領域のROI(グレースケール)
        @ param4[in] mask                   上記領域のマスク
        @ param5[in] threshold              閾値
        @ param6[in] abso                   画像における絶対座標で返すか返さないか
        @ param[out] iris_points(np.array)  虹彩座標リスト
        """
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

    def makeIrisMask(self, xpoint, ypoint, gray_img, eye_mask):
        """大津の二値化による瞳孔マスク作成
        @ param1[in]  xpoint                目の輪郭のX座標リスト
        @ param2[in]  ypoint                目の輪郭のY座標リスト
        @ param3[in]  gray_img              入力画像のグレースケール画像
        @ param4[in]  eye_mask              目の領域のマスク
        @ param1[out] iris_mask             虹彩マスク
        """
        top, bottom, left, right = extract.cutArea(xpoint, ypoint)
        eye_roi = gray_img[top:bottom, left:right]
        eye_mask_roi = eye_mask[top:bottom, left:right]
        iris_mask = np.zeros((eye_mask_roi.shape[0], eye_mask_roi.shape[1], 1), dtype=np.uint8)
        # 目領域内の輝度値取得
        ret, eye_luminance = self.defineThreshold(eye_roi, eye_mask_roi, rate=0.2)
        # 大津の二値化による閾値設定
        threshold = otsu.threshOtsu(eye_luminance)
        print("Otsu threshold : " , threshold)
        iris_points = self.detectIris(left, top, eye_roi, eye_mask_roi, threshold, abso=False)

        for i in range(len(iris_points)):
            iris_mask[iris_points[i][1]][iris_points[i][0]] = 255

        return iris_mask
