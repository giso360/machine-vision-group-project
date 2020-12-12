import cv2 
from sklearn import tree 
import numpy as np 

'''
    EXAMPLE
    -------------------------------
    ......attributes....... | CLASS
    -------------------------------
    overcast, wind, humidity, play 
    ['sunny', 'true', 'high'] 'yes'
    ['rainy', 'false','high'] 'no'
    -------------------------------
    training_X = [ ['sunny', 'true', 'high'],['rainy', 'false','high'] ]
    training_Y = ['yes', 'no']
    -------------------------------
    -------------------------------
    Similarly:
    bin1 px, bin2 px, ... | CLASS
    training_X = [[1000,0,0,0,12456], [2000,0,0,0,13456], ...]
'''

# if you change the lists with the images, 
# change the class values as well 
training = ['1.png','2.png','3.png','7.png'] 
testing = ['5.png','6.png','4.png']

training_X = [] # to append lists
training_Y = ["one","one","two","three"] # class outcomes/values 

testing_X = [] 
testing_Y = ["two","one","two"]

for i in range (len(training)):
    img = cv2.imread(training[i],0)
    hist,bins = np.histogram(img.flatten(),5, [0,256] ) 
    training_X.append(hist)

for i in range (len(testing)):
    img = cv2.imread(testing[i],0)
    hist,bins = np.histogram(img.flatten(),5, [0,256] ) 
    testing_X.append(hist)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_X, training_Y) # training
res = clf.predict(testing_X) # testing
print (res, "\n compared with \n, ",testing_Y) 









