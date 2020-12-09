import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('C12.jpg', 0)
nothing_img = cv2.imread('nothing12.jpg', 0)


# Image grayscale specs
print(img.shape)
print(img.size)

# histogram
bin_number = 256
hist, bins = np.histogram(img, bin_number, [0, bin_number])
plt.plot(hist)
plt.show()

no_back = cv2.subtract(nothing_img, img)
plt.hist(no_back, 256, [0, 256])
plt.show()

cv2.createTrackbar('T', 'image', 0, 255, no_back)

# create loop with a break condition
while 1:
    # get trackbar position
    current = cv2.getTrackbarPos('T', 'image')

    # do some operation
    ret, thresh1 = cv2.threshold(img, current, 255, cv2.THRESH_BINARY)

    # view result
    result = np.hstack((img, thresh1))
    cv2.imshow('image', result)

    # if key = "escape"
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

binary_thresh = 30

ret, thresh = cv2.threshold(no_back, binary_thresh, 255, cv2.THRESH_BINARY)
result = np.hstack((img, nothing_img, no_back))

# resized_image = cv2.resize(result, (500, 500))
# cv2.imshow('1000x1000', resized_image)
cv2.imshow('img', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
