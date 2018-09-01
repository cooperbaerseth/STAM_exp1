from keras.datasets import mnist
import numpy as np


def accuracy_eval(progress):
    correct = 0.0
    for i in range(0, progress):
        if x_clusterInd[i] == y_train[i]:
            correct = correct + 1

    return correct/progress


alpha = 0.05

#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
centroids = np.zeros((28*28, 10))
centroidIndexs = np.zeros(10)
x_clusterInd = np.zeros(x_train.shape[0])


###pick which instances will be centroids

#first of each class in train is centroid
for i in range(0,10):
    j = 0
    while(y_train[j] != i):
        j = j+1
    centroids[:,i] = x_train[j].flatten()
    centroidIndexs[i] = j
    x_clusterInd[j] = -1    #-1 indicates it is a centroid, not to be used in training

    print "index: " + str(j) + "\n digit: " + str(y_train[j])

#cluster
for i in range(0, x_train.shape[0]):     #over all instances in training set (60k)
    #don't use centroids
    if i in centroidIndexs:
        continue

    xi = x_train[i].flatten()

    # find closest centroid
    smallest = float("inf")    #holds distance through iterations
    for j in range(0,10):
        if np.linalg.norm(xi - centroids[:, j]) < smallest:
            smallest = np.linalg.norm(xi - centroids[:, j])
            x_clusterInd[i] = j

    #adjust centroid according to instance
    #centroids[:, j] = centroids[:, j] + (alpha*xi)
    centroids[:, j] = (1-alpha) * centroids[:, j] + (alpha * xi)

    #evaluate accuracy occasionally
    if i % 500 == 0:
        accu = accuracy_eval(i)
        print "Accuracy: " + str(accu*100) + "%"









print("done")