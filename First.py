import cv2
import numpy as np
from matplotlib import pyplot as plt


def connected_component_label(path):
    # Getting the input image
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Applying cv2.connectedComponents()
    num_labels, labels = cv2.connectedComponents(img)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()

    # Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()




def funcCan(self):
    thresh11 = cv2.getTrackbarPos('thresh1', 'canny')
    thresh22 = cv2.getTrackbarPos('thresh2', 'canny')
    edge = cv2.Canny(no_back, thresh11, thresh22)
    cv2.imshow('canny', edge)


# Loading an ASL related image
print("read image for gesture")
img = cv2.imread('./data/develop/B12.jpg', 0)
print("read image for background")
nothing_img = cv2.imread('./data/develop/nothing12.jpg', 0)

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
print("BEGIN subtract")
no_back = cv2.subtract(nothing_img, img)
print("END subtract")
# plt.hist(no_back, 256, [0, 256])
plt.show()
#
# Binary transformation of the image to black and white
binary_thresh = 10
print(no_back.shape)
no_back = np.where(no_back < 30, 0, no_back)
print(no_back.shape)
ret, thresh = cv2.threshold(no_back, binary_thresh, 255, cv2.THRESH_BINARY)
# Connected components
no_back2 = no_back
num_labels, labels = cv2.connectedComponents(no_back)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# Converting cvt to BGR
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
plt.show()


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
