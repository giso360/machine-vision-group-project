import cv2
import numpy as np
from matplotlib import pyplot as plt


def funcCan(self):
    thresh11 = cv2.getTrackbarPos('thresh1', 'canny')
    thresh22 = cv2.getTrackbarPos('thresh2', 'canny')
    edge = cv2.Canny(img, thresh11, thresh22)
    cv2.imshow('canny', edge)


# Loading an ASL related image
img = cv2.imread('A12.jpg', 0)
nothing_img = cv2.imread('nothing12.jpg', 0)

# Image shape/size in grayscale
# print(img.shape)
# print(img.size)
#
# # histogram
# bin_number = 256
# hist, bins = np.histogram(img, bin_number, [0, bin_number])
# plt.plot(hist)
# plt.show()
#
# Subtracting the background to isolate the hand
no_back = cv2.subtract(nothing_img, img)
plt.hist(no_back, 256, [0, 256])
plt.show()
#
# Binary transformation of the image to black and white
binary_thresh = 10
ret, thresh = cv2.threshold(no_back, binary_thresh, 255, cv2.THRESH_BINARY)

cv2.namedWindow('canny')
# create trackbars for given thresholds
thresh1 = 255
thresh2 = 101
cv2.createTrackbar('thresh1', 'canny', thresh1, 255, funcCan)
cv2.createTrackbar('thresh2', 'canny', thresh2, 255, funcCan)

# Call the
funcCan(self=0)

Canny_edge = cv2.Canny(no_back, thresh1, thresh2)
result = np.hstack((img, nothing_img, no_back, thresh, Canny_edge))
cv2.imshow('try', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
