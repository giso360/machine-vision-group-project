import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('C:/Python37/Lib/site-packages/cv2/data/haarcascade_smile.xml')

img = cv2.imread('pic2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    mouth = mouth_cascade.detectMultiScale(roi_gray, 
                                               scaleFactor=1.7,
                                               minNeighbors=22,
                                               minSize=(25, 25),)
    for (mx,my,mw,mh) in mouth:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
        
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
