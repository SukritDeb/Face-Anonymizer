import os
import argparse
import cv2

def process_img(img, face_detector):
    H, W, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        img[y:y + h, x:x + w, :] = cv2.blur(img[y:y + h, x:x + w, :], (30, 30))
    return img

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while ret:
    frame = process_img(frame, face_detector)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
cap.release()
cv2.destroyAllWindows()
