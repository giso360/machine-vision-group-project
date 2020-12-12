import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.png',0)

# histogram on flattened image ([[],[]] => [])
hist,bins = np.histogram(img.flatten(), 256, [0,256])

#cdf = cumulative sum
cdf = hist.cumsum()
cdf_normalised = (cdf * hist.max()) / cdf.max()

#print (img.shape[0],img.shape[1]) # size x*y

#adding to plot
plt.plot(cdf_normalised, color = 'b')
plt.plot(hist, color = 'r')

#cosmetics for x,y ranges and legend
plt.xlim([0,256]) 
plt.ylim([0,hist.max()])
plt.legend(('cdf norm','hist'), loc = 'upper left')
plt.show()

cv2.imshow('lena1', img)
cv2.waitKey(0) 

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = ((cdf_m - cdf_m.min())*255) / ((img.shape[0]*img.shape[1]) - cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img] 
hist, bins = np.histogram(img2.flatten(), 256, [0,256])

cdf = hist.cumsum()
cdf_normalised = (cdf * hist.max()) / cdf.max()

#adding to plot
plt.plot(cdf_normalised, color = 'b')
plt.plot(hist, color = 'r')

#cosmetics for x,y ranges and legend
plt.xlim([0,256]) 
plt.ylim([0,hist.max()])
plt.legend(('cdf norm','hist'), loc = 'upper left')
plt.show()

cv2.imshow('lena2', img2)
cv2.waitKey(0) 

# the automated way 
img3 = cv2.imread('lena.png',0)
equalized = cv2.equalizeHist(img3)

# horizontal/vertical stack h/vstack
result = np.hstack((img3, equalized))

cv2.imshow('together', result )
cv2.waitKey(0) 

#contrast limited adaptive histogram equalization
img4 = cv2.imread('lena.png',0)

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (10,10))
clahe_image = clahe.apply(img4)
result = np.hstack((img4, clahe_image))
cv2.imshow('together', result )
cv2.waitKey(0) 

img5 = cv2.imread('lena.png',0)

clahe1 = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
clahe2 = cv2.createCLAHE(clipLimit = 99.0, tileGridSize = (8,8))

cl1 = clahe1.apply(img5)
cl2 = clahe2.apply(img5)

three = np.hstack((img5, cl1, cl2))
cv2.imshow('three', three)

width = int(three.shape[1] * 0.6)
height = int(three.shape[0] * 0.6)
dim = (width, height)
resized = cv2.resize(three,dim,interpolation=cv2.INTER_AREA)

cv2.imshow('resized', resized)
cv2.waitKey(0)













