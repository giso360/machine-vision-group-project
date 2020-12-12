import cv2
import numpy as np

# dummy function for argument to createTrackbar
def nothing(x):
    pass
    
img = cv2.imread('pic2.png')
img_gray = cv2.imread('pic2.png',0)
template = cv2.imread('template1.png',0)

# name window early - need to know where to place trackbar
cv2.namedWindow('image')

# value/variable to be tracked: 'T'
# in window 'image' 
# see starting point and range
cv2.createTrackbar('T','image',90,100,nothing)

while(1):
    # get position
    current = cv2.getTrackbarPos('T','image')
    # set between 0 and 1 
    current = float(current/100)

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    
    # check against your threshold == value from trackbar
    loc = np.where(res >= current)

    # (re-)initialise a COPY since you are painting the image
    temp = img.copy()
    
    # want to get (x,y) from an array for each Y/Xcoords = [...]
    for pt in zip(*loc[::-1]):
        # image, starting point, end, color, thickness  
        cv2.rectangle(temp, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imshow("image", temp)

    # ESC 
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()