import cv2
import numpy as np


def black_white_ratio(image):
    thresh_b_w = list(np.ravel(image))
    thresh_b = [e for e in thresh_b_w if e == 0]
    thresh_w = [e for e in thresh_b_w if e == 255]
    black_to_white_ratio_feature = len(thresh_b) / len(thresh_w)
    return black_to_white_ratio_feature


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

    largest_width = biggest_component[2]
    largest_height = biggest_component[3]
    largest_area = biggest_component[4]

    centroid_x = centroids[index_of_biggest_component][0]
    centroid_y = centroids[index_of_biggest_component][1]
    conn_comp_stats.append([largest_width, largest_height, largest_area, centroid_x, centroid_y])
    return conn_comp_stats


def contouring(image, img):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255, 255, 0), 2)
    #cv2.imshow("contours", img)

    hull = cv2.convexHull(contours)
    cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
    #cv2.imshow("hull+contours", img)

    return len(contours), len(hull)