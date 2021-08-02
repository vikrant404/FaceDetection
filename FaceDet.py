import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()
    faces = faceCascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





