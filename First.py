import cv2
import numpy as np
from matplotlib import pyplot as plt


def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imshow("Biggest component", img2)
    cv2.waitKey()




def our_connected_components(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]


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

# Extract Feature 1 (B:W ratio)
# Extract Feature 1 (B:W ratio)
# Extract Feature 1 (B:W ratio)
thresh_b_w = list(np.ravel(thresh))
thresh_b = [e for e in thresh_b_w if e == 0]
thresh_w = [e for e in thresh_b_w if e == 255]
black_to_white_ratio_feature = len(thresh_b) / len(thresh_w)

print(black_to_white_ratio_feature)

# Send to def


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
canny_edge = cv2.Canny(no_back, thresh1, thresh2)

# Extract Feature 2 (no of components from edge canny)
ret, labels = cv2.connectedComponents(canny_edge)
no_of_components_feature = ret

# Extract Feature 3 (length of all edges detected)
canny_edge_flat = list(np.ravel(canny_edge))
total_edge_length_feature = [e for e in canny_edge_flat if e == 255]
total_edge_length_feature = len(total_edge_length_feature)
print(total_edge_length_feature)

# Extract Features (from connectedComponentsWithStats -> no_of_components, [width, height, area, centroids - Of biggest component])
# Extract Feature 4 (from connectedComponentsWithStats) no_of_components
# Extract Feature 5 (from connectedComponentsWithStats) [width - Of biggest component])
# Extract Feature 6 (from connectedComponentsWithStats) [height Of biggest component])
# Extract Feature 7 (from connectedComponentsWithStats) [area - Of biggest component])
# Extract Feature 8 (from connectedComponentsWithStats) [centroids - Of biggest component])

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny_edge, connectivity=8)


print("ooo")
index_of_biggest_component = np.argsort(-stats[:,-1])[1]
print("ooo")
print(nb_components)
print("ooo")
print(stats)
print("ooo")
print(output)
print("ooo")
print(centroids)
indexes_group = np.argsort(stats[:, cv2.CC_STAT_AREA])
stats = stats[indexes_group]
biggest_component = stats[len(stats)-2]
width_bounding_box_biggest = biggest_component[2]
height_bounding_box_biggest = biggest_component[3]
area_bounding_box_biggest = biggest_component[4]
print(biggest_component)

x_centroid_of_biggest = centroids[index_of_biggest_component][0]
y_centroid_of_biggest = centroids[index_of_biggest_component][1]

# y_centroid_of_biggest = centroids[index_of_biggest_component][1]
print("pppp")
# print(x_centroid_of_biggest)
print()
print("pppp")
result = np.hstack((img, nothing_img, no_back, thresh, canny_edge))
cv2.imshow('try', result)


# Extract Feature 9 Contours - no_of contours






cv2.waitKey(0)
cv2.destroyAllWindows()

# #########################################

