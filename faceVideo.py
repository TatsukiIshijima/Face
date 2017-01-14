# -*- coding: utf-8 -*-
import numpy as np
import cv2
import dlib
import sys

import draw

if __name__=='__main__':

    draw = draw.draw()

    # 検出器等の準備
    facedetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    src = cv2.VideoCapture(0)

    if not src.isOpened():
        print("Can not open camera")
        sys.exit()

    src.set(3, 640) # 横サイズ
    src.set(4, 480) # 縦サイズ

    while(True):

        ret, frame = src.read()
        height, width = frame.shape[:2]
        draw_frame = frame.copy()

        """顔の検出"""
        faces, scores, types = facedetector.run(frame, 1)
        for i, face in enumerate(faces):
            top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
            if min(top, height - bottom - 1, left, width - right -1) < 0:
                continue
            cv2.rectangle(draw_frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.putText(draw_frame, "Scores : " + str(scores[i]), (10, i+15), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
            cv2.putText(draw_frame, "Orientation : " + str(types[i]), (10, i+30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))

            """顔の特徴点"""
            draw.drawFacePoint(draw_frame, predictor, face, line=True, point=True)

        cv2.imshow('Image', draw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    src.release()
    cv2.destroyAllWindows()
