import cv2
import numpy as np


def black_white_ratio(image):
    thresh_b_w = list(np.ravel(image))
    thresh_b = [e for e in thresh_b_w if e == 0]
    thresh_w = [e for e in thresh_b_w if e == 255]
    black_to_white_ratio_feature = len(thresh_b) / len(thresh_w)
    print("Feat1-B/W ratio: ", black_to_white_ratio_feature)
    return black_to_white_ratio_feature


def canny_edges_thresholds(self):
    thresh11 = cv2.getTrackbarPos('thresh1', 'canny')
    thresh22 = cv2.getTrackbarPos('thresh2', 'canny')
    edge = cv2.Canny(no_background, thresh11, thresh22)
    cv2.imshow('canny', edge)


def canny_edges(thresh_1, thresh_2, image):
    # Extract Feature 2 (Number of components from Canny edge detection)
    canny_edge = cv2.Canny(image, thresh_1, thresh_2)
    return_value, canny_labels = cv2.connectedComponents(canny_edge)
    no_of_components_feature = return_value

    # Extract Feature 3 (Number of edges detected)
    canny_edge_flat = list(np.ravel(canny_edge))
    total_edge_length_feature = [e for e in canny_edge_flat if e == 255]
    total_edge_length_feature = len(total_edge_length_feature)
    return canny_edge, no_of_components_feature, total_edge_length_feature


def connected_components_statistics(image, conn):
    conn_comp_stats = []
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=conn)

    index_of_biggest_component = np.argsort(-stats[:, -1])[1]
    indexes_group = np.argsort(stats[:, cv2.CC_STAT_AREA])

    stats = stats[indexes_group]
    biggest_component = stats[len(stats) - 2]
    print(biggest_component)

    largest_width = biggest_component[2]
    largest_height = biggest_component[3]
    largest_area = biggest_component[4]

    centroid_x = centroids[index_of_biggest_component][0]
    centroid_y = centroids[index_of_biggest_component][1]
    conn_comp_stats.append([largest_width, largest_height, largest_area, centroid_x, centroid_y])
    return conn_comp_stats


def contouring(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255, 255, 0), 2)
    cv2.imshow("contours", img)

    hull = cv2.convexHull(contours)
    cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
    cv2.imshow("hull+contours", img)

    print(len(contours), len(hull))
    return len(contours), len(hull)


# Loading an ASL related image, its equivalent in grayscale and the background
img = cv2.imread('C12.jpg')
gray_img = cv2.imread('C12.jpg', 0)
nothing_img = cv2.imread('nothing12.jpg', 0)

# Grayscale Image shape/size
print(img.shape)
print(img.size)

# Subtracting the background to isolate the hand
no_background = cv2.subtract(nothing_img, gray_img)

# Binary transformation of the image to black and white
binary_thresh = 10
no_background = np.where(no_background < 30, 0, no_background)
ret, binary_img = cv2.threshold(no_background, binary_thresh, 255, cv2.THRESH_BINARY)

# Feature 1 (Black/White pixels ratio)
feature_1 = black_white_ratio(binary_img)

# Canny Edge Detection

cv2.namedWindow('canny')

# Creating trackbars to help us determine the canny edge detection thresholds
cv2.createTrackbar('thresh1', 'canny', 0, 255, canny_edges_thresholds)
cv2.createTrackbar('thresh2', 'canny', 0, 255, canny_edges_thresholds)

# Call the function canny_edges to get a hold of good thresholds
canny_edges_thresholds(self=0)

thresh1 = 255
thresh2 = 101

# Features 2,3 Canny Edge # of components and # of edges
canny_edge_img, feature_2, feature_3 = canny_edges(thresh1, thresh2, no_background)

# Features 4-8  Stats of the largest connected component -> [Height,width,Area,Centroid coordinates]
connectivity = 8
connected_comp_stats = connected_components_statistics(canny_edge_img, connectivity)

# Features 9-10 Contouring features (Hull and Contours of the image)
Hull, contour = contouring(no_background)

result = np.hstack((gray_img, nothing_img, no_background, binary_img, canny_edge_img))
cv2.imshow('try', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
