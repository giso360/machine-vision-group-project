import cv2
import numpy as np
from matplotlib import pyplot as plt 

# img = cv2.imread('face.png', 0)
img = cv2.imread('../data/develop/D12.jpg', 0)

nothing_img = cv2.imread('../data/develop/nothing12.jpg', 0)
no_back = cv2.subtract(nothing_img, img)
binary_thresh = 10
# no_back = np.where(no_back < 30, 0, no_back)
ret, thresh = cv2.threshold(no_back, binary_thresh, 255, cv2.THRESH_BINARY)




img = cv2.Canny(no_back,100,200)
ret, labels = cv2.connectedComponents(img)
print(ret)

# Map component labels to hue val

# why 179?
label_hue = np.uint8(179*labels/np.max(labels))

#why 255?
#why ones_like?
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# set bg label to black
labeled_img[label_hue==0] = 0

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#cv2.imshow('check',labeled_img)
#cv2.waitKey(0)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(labeled_img,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()




# Contours

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)




