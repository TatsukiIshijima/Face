# -*- coding:utf-8 -*-
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import draw
import extract
import calculation
import iris
import pupil

draw = draw.draw()
extract = extract.extract()
calc = calculation.calculation()
iris = iris.iris()
pupil = pupil.pupil()

def convertXPointList(pointList, x):
    """座標点リストをX,Y座標リストへ変換
    @ param1[in] pointList      座標点リスト
    @ param2[in] x              X座標の場合は0、Y座標の場合は1を指定
    @ param[out] xPointList     変換座標
    """
    xPointList = []
    for i in range(len(pointList)):
        if x == 0:
            xPointList.append(pointList[i][x])
        elif x == 1:
            xPointList.append(pointList[i][x])
        else:
            print("x is out of range. x = 0 or 1")
            sys.exit()
    return xPointList

def getEyeLidXPoint(xPointList, flg):
    """瞼のX,Y座標リストを取得
    @ param1[in] xPointList     X or Y 座標リスト
    @ param2[in] flg            Trueは上瞼, Falseは下瞼
    @ param[out] eyelidPoint    瞼座標
    """
    eyeLidPoint = []
    # 上瞼
    if flg:
        for i in range(0, 4):
            eyeLidPoint.append(xPointList[i])
    # 下瞼
    else:
        for i in range(-3, 1):
            eyeLidPoint.append(xPointList[i])

    return eyeLidPoint

def getEyeLidCurvature(midPoint, xpoint, ypoint):
    """瞼の曲率算出
    @ param1[in]  midPoint              目頭と目尻の中点
    @ param2[in]  xpoint                目の輪郭のX座標リスト
    @ param3[in]  ypoint                目の輪郭のY座標リスト
    @ param1[out] eyeLid_upper_curva    中点のX座標における上瞼の曲率
    @ param2[out] eyeLid_lower_curva    中点のX座標における下瞼の曲率
    """
    eyeLid_upperX = getEyeLidXPoint(xpoint, flg=True)
    eyeLid_lowerX = getEyeLidXPoint(xpoint, flg=False)
    eyeLid_upperY = getEyeLidXPoint(ypoint, flg=True)
    eyeLid_lowerY = getEyeLidXPoint(ypoint, flg=False)
    eyeLid_upper_curva = calc.calcCurvature(eyeLid_upperX, eyeLid_upperY, midPoint[0])
    eyeLid_lower_curva = calc.calcCurvature(eyeLid_lowerX, eyeLid_lowerY, midPoint[0])
    return eyeLid_upper_curva, eyeLid_lower_curva

def getPupilPoint(xpoint, ypoint, gray_img, eye_mask, rate):
    """瞳孔検出
    @ param1[in]  xpoint                目の輪郭のX座標リスト
    @ param2[in]  ypoint                目の輪郭のY座標リスト
    @ param3[in]  gray_img              入力画像のグレースケール画像
    @ param4[in]  eye_mask              目の領域のマスク
    @ param5[in]  rate                  輝度比
    @ param1[out] cx                    重心X座標
    @ param2[out] cy                    重心Y座標
    @ param3[out] iris_hull             虹彩の凸包座標
    """
    # 目の領域抽出
    top, bottom, left, right = extract.cutArea(xpoint, ypoint)
    eye_roi = gray_img[top:bottom, left:right]
    eye_mask_roi = eye_mask[top:bottom, left:right]
    # 目の虹彩、瞳孔検出
    threshold = iris.defineThreshold(eye_roi, eye_mask_roi, rate)
    iris_points = iris.detectIris(left, top, eye_roi, eye_mask_roi, threshold)
    if len(iris_points) == 0:
        cx = None
        cy = None
        iris_hull = []
    else:
        cx, cy, iris_hull = pupil.detectPupil(iris_points)
    return cx, cy, iris_hull

if __name__ == '__main__':

    param = sys.argv

    # 検出器等の準備
    facedetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(param[1])
    height, width = image.shape[:2]
    gray_img = image.copy()
    draw_img = image.copy()
    eye_mask = np.zeros((height, width, 1), dtype=np.uint8)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    # dlib version 18.18だと引数が異なるので注意
    faces, scores, types = facedetector.run(image, 0)
    for i, face in enumerate(faces):
        top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        if min(top, height - bottom - 1, left, width - right -1) < 0:
            continue
        cv2.rectangle(draw_img, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.putText(draw_img, "Scores : " + str(scores[i]), (10, i+15), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        cv2.putText(draw_img, "Orientation : " + str(types[i]), (10, i+30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        draw.drawFacePoint(draw_img, predictor, face, line=True, point=True)

        # 鼻の側面の輪郭取得
        r_nose_contour, l_nose_contour = extract.getNoseSideContour(image, predictor, face)
        if len(r_nose_contour) != 3 and len(l_nose_contour) != 3:
            continue
        cv2.drawContours(draw_img, [r_nose_contour], -1, (255, 255, 0), -1)
        cv2.drawContours(draw_img, [l_nose_contour], -1, (0, 255, 255), -1)
        # 面積算出
        r_nose_area = cv2.contourArea(r_nose_contour)
        l_nose_area = cv2.contourArea(l_nose_contour)
        cv2.putText(draw_img, "R Nose Area : " + str(r_nose_area) + ", L Nose Aea : " + str(l_nose_area), (10, i+90), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))

        # 右目、左目の輪郭取得
        r_eye_contour, l_eye_contour = extract.getEyeContour(image, predictor, face)
        # 輪郭座標をX、Y座標リストに分割
        r_xpoint = convertXPointList(r_eye_contour, 0)
        r_ypoint = convertXPointList(r_eye_contour, 1)
        l_xpoint = convertXPointList(l_eye_contour, 0)
        l_ypoint = convertXPointList(l_eye_contour, 1)
        # 目の領域のマスク作成
        cv2.drawContours(eye_mask, [r_eye_contour], -1, (255, 255, 255), -1)
        cv2.drawContours(eye_mask, [l_eye_contour], -1, (255, 255, 255), -1)
        # 面積算出
        r_eye_area = cv2.contourArea(r_eye_contour)
        l_eye_area = cv2.contourArea(l_eye_contour)
        cv2.putText(draw_img, "R Area : " + str(r_eye_area) + ", L Aea : " + str(l_eye_area), (10, i+45), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))

        if len(r_eye_contour) != 6:
            continue
        # 右目の瞼の曲率計算
        r_midPoint = calc.calcMidPoint(r_eye_contour[0], r_eye_contour[3])
        r_lid_upper, r_lid_lower = getEyeLidCurvature(r_midPoint, r_xpoint, r_ypoint)
        cv2.putText(draw_img, "(R)Upper Lid Curva : " + str(r_lid_upper) + ", (R)Lower Lid Curva : " + str(r_lid_lower), (10, i+60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        # 右目の瞳孔検出
        R_cx, R_cy, R_iris_hull = getPupilPoint(r_xpoint, r_ypoint, gray_img, eye_mask, 0.2)
        cv2.drawContours(draw_img, [R_iris_hull], -1, (255, 255, 0), 1)
        cv2.circle(draw_img, (R_cx, R_cy), 2, (0, 255, 255), -1)

        if len(l_eye_contour) != 6:
            continue
        # 左目の瞼の曲率計算
        l_midPoint = calc.calcMidPoint(l_eye_contour[0], l_eye_contour[3])
        l_lid_upper, l_lid_lower = getEyeLidCurvature(l_midPoint, l_xpoint, l_ypoint)
        cv2.putText(draw_img, "(L)Upper Lid Curva : " + str(l_lid_upper) + ", (L)Lower Lid Curva : " + str(l_lid_lower), (10, i+75), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        # 左目の瞳孔検出
        L_cx, L_cy, L_iris_hull = getPupilPoint(l_xpoint, l_ypoint, gray_img, eye_mask, 0.2)
        cv2.drawContours(draw_img, [L_iris_hull], -1, (255, 255, 0), 1)
        cv2.circle(draw_img, (L_cx, L_cy), 2, (0, 255, 255), -1)

    cv2.imshow("Image", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
