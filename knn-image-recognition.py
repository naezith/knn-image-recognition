'''
    @author: naezith
'''

import cv2
import numpy as np
from os import walk
from sklearn.neighbors import KNeighborsClassifier

# Class names 0,         1,                 2
classes = ['danger-warning', 'park-stop', 'traffic']

danger_traffic = [0, 2]
danger_park = [0, 1]
all_groups = [0, 1, 2]

# Checks if bottom left pixel is red
color_threshold = 200
def bottomLeftIsRed(img):
    col = img[int(0.99 * img.shape[0]), int(0.05 * img.shape[1])]
    return int(col[2] > color_threshold and col[1] < color_threshold and col[0] < color_threshold)

# Checks if bottom left pixel is red
def bottomLeftIsBlue(img):
    col = img[int(0.94 * img.shape[0]), int(0.07 * img.shape[1])]
    return int(col[2] < color_threshold and col[1] < color_threshold and col[0] > color_threshold)

# Checks if bottom left pixel is red
def knn(k, val, points):
    # Make a list of distances
    dists = []
    for c, point in points: 
        dists.append([abs(point[0] - val[0]) + abs(point[1] - val[1]), c])
        
    # Sort the distances to get closest ones in front 
    dists = sorted(dists, key = lambda d: d[0])
    
    # Count k closest labels    
    label_counts = [[0, 0], [1, 0], [2, 0]]
    for i in range(0, k):
        label_counts[dists[i][1]][1] += 1
    
    # Sort reversed to find the most counted one, return the label of it
    return sorted(label_counts, key = lambda d: d[1], reverse=True)[0][0]
# sklearn knn
def knn_sklearn(k, val, points):
    neigh = KNeighborsClassifier(n_neighbors = k)
    # points to classes
    
    X = np.array([row[1] for row in points]);
    y = np.array([row[0] for row in points]);
    
    neigh.fit(X, y);
    
    return neigh.predict(np.array([val]))[0] 

# Checks if bottom left pixel is red
def getConfusionMatrices(knn_func, test_images, points):
    # Fill the confusion matrices, first 3 for k=1,3,5 and 4th one is for best of k=1,3,5
    results = [] 
    for c, img in test_images:
        knn_results = []
        
        knn_counts = [0, 0, 0]
        i = 0
        # KNN for k=1, 3, 5
        for k in range(1, 6, 2):
            result = [c, knn_func(k, [bottomLeftIsRed(img), bottomLeftIsBlue(img)], points)]
            knn_results.append(result)
            knn_counts[result[1]] += 1; 
            
        # Find the max
        max_idx = 0
        for i in range(1, 3):
            if knn_counts[i] > max_idx: max_idx = i;
                            
        knn_results.append([c, i])
        
        # Add it to the results list
        results.append(knn_results)
        
    # Transpose it to have separate confusion matrices
    return zip(*results)


# Print Confusion Matrices
def printConfusionMatrices(matrices):
    k = 1
    for row in matrices:
        if k == 7: print ('Most -> ')
        else: print ('K ='), k, ('->')
    
        for val in row:
            print ('{:4}').format(val),
        print
        k += 2

# Filter by classes
def filterByClass(data_list, class_list):
    return [d for d in data_list if d[0] in class_list]

# Test classes
def test(class_list): 
    print ('\n\n')
    for c in class_list: print (classes[c]),
    print ('\nCustom KNN')
    printConfusionMatrices(getConfusionMatrices(knn, filterByClass(test_images, class_list), filterByClass(points, class_list)))
    print ('\nsklearn KNN')
    printConfusionMatrices(getConfusionMatrices(knn_sklearn, filterByClass(test_images, class_list), filterByClass(points, class_list)))


# Load training data
points = []
for c in range(0, len(classes)):
    folder = 'signs/train/' + classes[c]
    for (dirpath, dirnames, filenames) in walk(folder):
        for img_name in filenames:
            img = cv2.imread(folder + '/' + img_name, cv2.IMREAD_COLOR)
            points.append([c, [bottomLeftIsRed(img), bottomLeftIsBlue(img)]])

# Load test images
test_images = []
for c in range(0, len(classes)):
    folder = 'signs/test/' + classes[c]
    for (dirpath, dirnames, filenames) in walk(folder):
        for img_name in filenames:
            test_images.append([c, cv2.imread(folder + '/' + img_name, cv2.IMREAD_COLOR)])


# Test for different classes
test(all_groups)
test(danger_park)
test(danger_traffic)