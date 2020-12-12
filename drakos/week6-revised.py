import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('CA.png',0)
cv2.imshow('original', img)
cv2.waitKey(0)
cv2.destroyAllWindows() # all active windows 

# one way
#hist, bins = np.histogram(img.flatten(),5,[0,256])
#plt.plot(hist, color = 'r')

# another way 
plt.hist(img.flatten(), bins = 5)
plt.show()

thresholded_img = img.copy() 

for y in range (0, img.shape[0]):
    for x in range (0, img.shape[1]):
        if img[y,x] > 127:
            thresholded_img[y,x] = 255
        else:
            thresholded_img[y,x] = 0
            
together = np.hstack((img, thresholded_img))
cv2.imshow('image + B&W', together) 
cv2.waitKey(0)
cv2.destroyWindow('image + B&W') # specific window
    
img = cv2.imread('CA.png') # color     
rgb_img = img.copy()

flag = 0 # looking for black

for i in range (0, thresholded_img.shape[0]):

    if flag == 255:
        break 
        
    for j in range (0, thresholded_img.shape[1]):
        if thresholded_img[i,j] == 0: 
            starting_Y, starting_X = i, j 
            flag = 255 # next: looking for white 
            break 
        else: 
            rgb_img[i,j] = [255,0,0]
        
        res_img = np.hstack((img, rgb_img))

        cv2.imshow('CA', res_img)
        cv2.waitKey(1)

print (starting_Y, starting_X)
cv2.waitKey(0)

for j in range (starting_X, thresholded_img.shape[1]):
    if thresholded_img[starting_Y, j] == 255:
        ending_X = j 
        break 
    else:
        rgb_img[starting_Y, j] = [0,0,255]
        res_img = np.hstack((img, rgb_img))
        
        cv2.waitKey(1) # place wherever appropriate
        cv2.imshow('CA', res_img)

print (ending_X)

# paint pixel + surrounding ones 
img[starting_Y+1, starting_X+1] = [0,0,255]
img[starting_Y-1, starting_X+1] = [0,0,255]
img[starting_Y+1, starting_X] = [0,0,255]
img[starting_Y-1, starting_X] = [0,0,255]
img[starting_Y, starting_X] = [0,0,255]
img[starting_Y, starting_X+1] = [0,0,255]
img[starting_Y, starting_X-1] = [0,0,255]
img[starting_Y+1, starting_X-1] = [0,0,255]
img[starting_Y-1, starting_X-1] = [0,0,255]

res_img = np.hstack((img, rgb_img))
cv2.imshow('CA', res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()









       
    








    
    
    
    
    
    
    
    
    
    
    
    