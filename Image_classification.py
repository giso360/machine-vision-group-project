import os
from util import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.model_selection import train_test_split


def getlabel(file):
    return file.split('/')[-1][0]


df_data = []
background = './nothing12.jpg'
back_img = cv2.imread(background, 0)

dir_name = './A'
files = [dir_name + "/" + file for file in os.listdir(dir_name)]
labels = [e.split('/')[-1][0] for e in files]

for file in os.listdir(dir_name):
    img = cv2.imread(dir_name + "/" + file)
    gray_img = cv2.imread(dir_name + "/" + file, 0)

    no_background = cv2.subtract(back_img, gray_img)

    binary_thresh = 10
    no_background = np.where(no_background < 30, 0, no_background)
    ret, binary_img = cv2.threshold(no_background, binary_thresh, 255, cv2.THRESH_BINARY)

    thresh1 = 255
    thresh2 = 101

    # Feature 1 (Black/White pixels ratio)
    feature_1 = black_white_ratio(binary_img)

    # Features 2,3 Canny Edge # of components and # of edges
    canny_edge_img, feature_2, feature_3 = canny_edges(thresh1, thresh2, no_background)

    # Features 4-8  Stats of the largest connected component -> [Height,width,Area,Centroid coordinates]
    connectivity = 8
    connected_comp_stats = connected_components_statistics(canny_edge_img, connectivity)

    # Features 9-10 Contouring features (Hull and Contours of the image)
    hull, contour = contouring(no_background, img)

    x = [item for sublist in connected_comp_stats for item in sublist]
    features = [feature_1, feature_2, feature_3, hull, contour]
    for item in x:
        features.append(item)
    features.append(getlabel(dir_name + "/" + file))
    df_data.append(features)

column_index = ['B/W ratio', '#Canny_Components', '#Canny_Edges', 'CC_Height', 'CC_Width', 'CC_Area', 'CC_centroid_X',
                'CC_centroid_Y', 'Contour', 'Hull', 'Label']
df = pd.DataFrame(data=df_data, columns=column_index)

X = df.drop("Label", 1)
X = X.astype(float)

y = df['Label']
y.replace(to_replace='A', value=0, inplace=True)
y.replace(to_replace='B', value=1, inplace=True)
y.replace(to_replace='C', value=2, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clfDT = tree.DecisionTreeClassifier()
print('asd')
clfDT.fit(x_train, y_train)
y_test_pred_DT = clfDT.predict(x_test)

print(precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro'))
