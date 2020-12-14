import cv2
import numpy as np
from matplotlib import pyplot as plt


img_path = "./data/develop/D12.jpg"
img = cv2.imread(img_path)
img_0 = cv2.imread(img_path, 0)
back = cv2.imread("./data/develop/nothing12.jpg", 0)
img_0 = cv2.subtract(back, img_0)


cv2.imshow('palm image',img)

ret, thresh = cv2.threshold(img_0, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv2.contourArea(x))
cv2.drawContours(img, [contours], -1, (255,255,0), 2)
cv2.imshow("contours", img)

hull = cv2.convexHull(contours)
cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
cv2.imshow("hull", img)

hull = cv2.convexHull(contours, returnPoints=False)
defects = cv2.convexityDefects(contours, hull)

#
if defects is not None:
  cnt = 0
for i in range(defects.shape[0]):  # calculate the angle
  s, e, f, d = defects[i][0]
  start = tuple(contours[s][0])
  end = tuple(contours[e][0])
  far = tuple(contours[f][0])
  a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
  b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
  c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
  angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
  if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
    cnt += 1
    cv2.circle(img, far, 4, [0, 0, 255], -1)
if cnt > 0:
  cnt = cnt+1
cv2.putText(img, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#
cv2.imshow('final_result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(img_0, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

print("Total Number of Contours found =", len(contours))
print("Total Number of hull found =", len(hull))
# print("hierarchy is: \n", hierarchy)
