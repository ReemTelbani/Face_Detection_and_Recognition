import cv2
from pip._vendor.distlib.compat import raw_input
import numpy as np
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from pip._vendor.distlib.compat import raw_input

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
i = 0
name = raw_input('enter your id')
while True:
    ret, im = cam.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    (width, height) = (130, 100)

    for(x, y, w, h) in faces:
        i = i+1
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]  # cropping faces
        face_resize = cv2.resize(roi_gray, (width, height))
        roi_color = im[y:y + h, x:x + w]
        cv2.imwrite("dataSet/face-" + name + '.' + str(i) + ".jpg", face_resize)

        cv2.imshow('im', face_resize)
        cv2.waitKey(100)
    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break

print("capturing done")