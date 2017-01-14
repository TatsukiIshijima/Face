# -*- coding:utf-8 -*-
import dlib
import cv2
import numpy as np

class draw:
    """描画クラス"""

    def drawFacePoint(self, image, predictor, face, line, point):
        """顔の特徴点描画
        @ param1[in] image              描画する画像
        @ param2[in] predictor          予測器
        @ param3[in] face               検出した顔領域
        @ param4[in] line               線で描画
        @ param5[in] point              点で描画
        """
        shape = predictor(image, face)
        for i in range(shape.num_parts):
            if line:
                # 輪郭
                if i >= 0 and i <= 15:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                # 右眉
                elif i >= 17 and i <= 20:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                # 左眉
                elif i >= 22 and i <= 25:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                # 鼻筋
                elif i >= 27 and i <= 29:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                # 鼻
                elif i >= 31 and i <= 34:
                    cv2.line(image, (shape.part(30).x, shape.part(30).y), (shape.part(31).x, shape.part(31).y), (0, 255, 0), 1)
                    cv2.line(image, (shape.part(30).x, shape.part(30).y), (shape.part(35).x, shape.part(35).y), (0, 255, 0), 1)
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                # 右目
                elif i >= 36 and i <= 40:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                    cv2.line(image, (shape.part(36).x, shape.part(36).y), (shape.part(41).x, shape.part(41).y), (0, 255, 0), 1)
                # 左目
                elif i >= 42 and i <= 46:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                    cv2.line(image, (shape.part(42).x, shape.part(42).y), (shape.part(47).x, shape.part(47).y), (0, 255, 0), 1)
                # 口
                elif i >= 48 and i <= 58:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                    cv2.line(image, (shape.part(48).x, shape.part(48).y), (shape.part(59).x, shape.part(59).y), (0, 255, 0), 1)
                elif i >= 60 and i <= 66:
                    cv2.line(image, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0, 255, 0), 1)
                    cv2.line(image, (shape.part(60).x, shape.part(60).y), (shape.part(67).x, shape.part(67).y), (0, 255, 0), 1)

            if point:
                cv2.circle(image, (shape.part(i).x, shape.part(i).y), 2, (255, 0, 0), -1)
