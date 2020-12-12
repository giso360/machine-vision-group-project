import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math 

# thresholds
img = cv2.imread('randomDude.png',0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['R', '1', '2', '3', '4', '5']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# instead of this, we can use nested hstack/vstack
for i in range (6):
    plt.subplot (2,3,i+1)
    plt.imshow(images[i],cmap='gray') # color map to gray
    plt.title(titles[i])
plt.show()
    
############################################## 

# 1D gaussian kernel creation
sigma = 1
step = 1
array = []

                # [from, to)
# for x in range (-3,4,step):

# can use this for float steps
for x in np.arange (-2,3,step):
    value = math.exp(-(x**2)/2*(sigma**2))/math.sqrt(2*math.pi*(sigma**2))
    array.append(value)
print(array)

array = np.array(array)
divided = np.round(array/array.min())
print (divided)




