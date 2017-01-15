# -*- coding:utf-8 -*-
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

import draw
import extract
import calculation
import iris
import pupil
import otsu

draw = draw.draw()
extract = extract.extract()
calc = calculation.calculation()
iris = iris.iris()
pupil = pupil.pupil()
otsu = otsu.otsu()

matplotlib.rcParams['font.size'] = 8

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
    @ param1[out] eye_roi               目領域のROI
    @ param2[out] eye_luminance         輝度値リスト
    @ param3[out] iris_mask             虹彩マスク
    @ param4[out] cx                    重心X座標
    @ param5[out] cy                    重心Y座標
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
        cx, cy, iris_hull = pupil.detectPupil(iris_points)
    return eye_roi, eye_luminance, iris_mask, cx, cy, iris_hull

def makePupilMask(xpoint, ypoint, gray_img, eye_mask):
    """大津の二値化による瞳孔マスク作成
    """
    top, bottom, left, right = extract.cutArea(xpoint, ypoint)
    eye_roi = gray_img[top:bottom, left:right]
    eye_mask_roi = eye_mask[top:bottom, left:right]
    pupil_mask = np.zeros((eye_mask_roi.shape[0], eye_mask_roi.shape[1], 1), dtype=np.uint8)
    # 目領域内の輝度値取得
    ret, eye_luminance = iris.defineThreshold(eye_roi, eye_mask_roi, rate=0.2)
    # 大津の二値化による閾値設定
    threshold = otsu.threshOtsu(eye_luminance)
    print("Otsu threshold : " , threshold)
    iris_points = iris.detectIris(left, top, eye_roi, eye_mask_roi, threshold, abso=False)

    for i in range(len(iris_points)):
        pupil_mask[iris_points[i][1]][iris_points[i][0]] = 255

    return pupil_mask

def morpho(mask, turn, iteration):
    if turn == True:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=iteration)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=iteration)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=iteration)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=iteration)

    return mask

if __name__ == '__main__':

    param = sys.argv

    # 検出器等の準備
    facedetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(param[1])
    #savepath = param[2]
    #image = cv2.resize(image, (480, 240))
    height, width = image.shape[:2]
    gray_img = image.copy()
    draw_img = image.copy()
    draw_otsu_img = image.copy()
    eye_mask = np.zeros((height, width, 1), dtype=np.uint8)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    # dlib version 18.18だと引数が異なるので注意
    faces, scores, types = facedetector.run(image, 0)
    if len(faces) == 0:
        print("Can not detect face")
        sys.exit()
    for i, face in enumerate(faces):
        top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        if min(top, height - bottom - 1, left, width - right -1) < 0:
            continue
        cv2.rectangle(draw_img, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.rectangle(draw_otsu_img, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.putText(draw_img, "Scores : " + str(scores[i]), (10, i+15), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        cv2.putText(draw_img, "Orientation : " + str(types[i]), (10, i+30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        draw.drawFacePoint(draw_img, predictor, face, line=True, point=True)
        draw.drawFacePoint(draw_otsu_img, predictor, face, line=True, point=True)

        # 顔の輪郭取得
        face_contour = extract.getFaceContour(image, predictor, face)
        if len(face_contour) != 17:
            continue
        # 輪郭から楕円近似
        #ellipse = cv2.fitEllipse(face_contour)
        #cv2.ellipse(draw_img, ellipse, (0, 255, 0), 1)

        # 鼻の側面の輪郭取得
        r_nose_contour, l_nose_contour = extract.getNoseSideContour(image, predictor, face)
        if len(r_nose_contour) != 3 and len(l_nose_contour) != 3:
            continue
        cv2.drawContours(draw_img, [r_nose_contour], -1, (255, 255, 0), 1)
        cv2.drawContours(draw_img, [l_nose_contour], -1, (0, 255, 255), 1)
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
        R_eye_ROI, R_eye_lumi, R_iris_mask1, R_cx, R_cy, R_iris_hull = getPupilPoint(r_xpoint, r_ypoint, gray_img, eye_mask, 0.25)
        R_iris_mask2 = makePupilMask(r_xpoint, r_ypoint, gray_img, eye_mask)
        if len(R_iris_hull) != 0:
            cv2.drawContours(draw_img, [R_iris_hull], -1, (255, 255, 0), 1)
            print("Number of R_iris_hull point : ", len(R_iris_hull))
        else:
            print("Can not detect R iris")

        if  R_cx != None and R_cy != None:
            cv2.circle(draw_img, (R_cx, R_cy), 2, (0, 255, 255), -1)
        else:
            print("Can not detect R pupil")

        if len(l_eye_contour) != 6:
            continue
        # 左目の瞼の曲率計算
        l_midPoint = calc.calcMidPoint(l_eye_contour[0], l_eye_contour[3])
        l_lid_upper, l_lid_lower = getEyeLidCurvature(l_midPoint, l_xpoint, l_ypoint)
        cv2.putText(draw_img, "(L)Upper Lid Curva : " + str(l_lid_upper) + ", (L)Lower Lid Curva : " + str(l_lid_lower), (10, i+75), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        # 左目の瞳孔検出
        L_eye_ROI, L_eye_lumi, L_iris_mask1, L_cx, L_cy, L_iris_hull = getPupilPoint(l_xpoint, l_ypoint, gray_img, eye_mask, 0.25)
        L_iris_mask2 = makePupilMask(l_xpoint, l_ypoint, gray_img, eye_mask)
        if len(L_iris_hull) != 0:
            cv2.drawContours(draw_img, [L_iris_hull], -1, (255, 255, 0), 1)
            print("Number of L_iris_hull point : ", len(L_iris_hull))
        else:
            print("Can not detect L iris")

        if L_cx != None and L_cy != None:
            cv2.circle(draw_img, (L_cx, L_cy), 2, (0, 255, 255), -1)
        else:
            print("Can not detect L iris")

    """
    #cv2.imwrite(savepath + "/Result.png", draw_img)
    #cv2.imwrite(savepath + "/Mask.png", eye_mask)
    #cv2.imwrite(savepath + "/R_EYE_ROI.png", R_eye_ROI)
    #cv2.imwrite(savepath + "/L_EYE_ROI.png", L_eye_ROI)
    """

    fig = plt.figure(figsize=(14, 7))
    ax_R_ROI = fig.add_subplot(3,4,1)
    ax_L_ROI = fig.add_subplot(3,4,2)
    ax_R_ROI_hist = fig.add_subplot(3,4,5)
    ax_L_ROI_hist = fig.add_subplot(3,4,6)
    ax_result = fig.add_subplot(3,4,7)
    ax_result_otsu = fig.add_subplot(3,4,8)
    ax_R_iris1 = fig.add_subplot(3,4,9)
    ax_R_iris2 = fig.add_subplot(3,4,10)
    ax_L_iris1 = fig.add_subplot(3,4,11)
    ax_L_iris2 = fig.add_subplot(3,4,12)

    ax_R_ROI.set_title("R eye")
    ax_L_ROI.set_title("L eye")
    ax_result.set_title("Result(Ver.Rate)")
    ax_result_otsu.set_title("Result(Ver.Otsu)")
    ax_R_ROI_hist.set_title("R eye Hist")
    ax_L_ROI_hist.set_title("L eye Hist")
    ax_R_iris1.set_title("R iris mask(Ver.Rate)")
    ax_R_iris2.set_title("R iris mask(Ver.Otsu)")
    ax_L_iris1.set_title("L iris mask(Ver.Rate)")
    ax_L_iris2.set_title("L iris mask(Ver.Otsu)")

    ax_R_ROI.imshow(R_eye_ROI, cmap=plt.cm.gray)
    ax_L_ROI.imshow(L_eye_ROI, cmap=plt.cm.gray)
    ax_result.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    ax_result_otsu.imshow(cv2.cvtColor(draw_otsu_img, cv2.COLOR_BGR2RGB))

    R_iris_mask1 = cv2.bitwise_and(R_eye_ROI, R_eye_ROI, mask=R_iris_mask1)
    R_iris_mask2 = cv2.bitwise_and(R_eye_ROI, R_eye_ROI, mask=R_iris_mask2)
    L_iris_mask1 = cv2.bitwise_and(L_eye_ROI, L_eye_ROI, mask=L_iris_mask1)
    L_iris_mask2 = cv2.bitwise_and(L_eye_ROI, L_eye_ROI, mask=L_iris_mask2)
    ax_R_iris1.imshow(R_iris_mask1, cmap=plt.cm.gray)
    ax_R_iris2.imshow(R_iris_mask2, cmap=plt.cm.gray)
    ax_L_iris1.imshow(L_iris_mask1, cmap=plt.cm.gray)
    ax_L_iris2.imshow(L_iris_mask2, cmap=plt.cm.gray)

    ax_R_ROI_hist.set_ylabel("Number of Pixels")
    ax_R_ROI_hist.set_xlabel("Luminance value")
    ax_R_ROI_hist.set_xlim([0, 256])
    ax_R_ROI_hist.hist(R_eye_lumi, 256, [0, 256], color='blue')
    R_hist_ymin, R_hist_ymax = ax_R_ROI_hist.get_ylim()

    ax_R_ROI_hist.set_ylabel("Number of Pixels")
    ax_L_ROI_hist.set_xlabel("Luminance value")
    ax_L_ROI_hist.set_xlim([0, 256])
    ax_L_ROI_hist.hist(L_eye_lumi, 256, [0, 256], color='blue')

    plt.show()
