from __future__ import print_function

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(True)


def accuracy_eval(progress):
    correct = 0.0
    for i in range(0, progress):
        if x_clusterInd[i] == y_train[i]:
            correct = correct + 1

    return correct/progress

def accuracy_eval_perDigit(progress):
    accu = np.zeros((3, 10))    #1st row: # correct (per class)
                                #2nd row: # seen (per class)
                                #3rd row: accuracy (per class)
                                #class indexed by position
    #get correct / total
    for i in range(0,progress):
        accu[1, y_train[i]] = accu[1, y_train[i]] + 1.0
        if x_clusterInd[i] == y_train[i]:
            accu[0, y_train[i]] = accu[0, y_train[i]] + 1.0

    #get accuracy per class
    for i in range(0, 10):
        accu[2,i] = accu[0,i] / accu[1,i]

    #print accuracy per class
    print("Per Class Accuracy: ")
    for i in range(0, 10):
        print(str(i) + ": " + str(round(accu[2,i], 3)) + "\t\t", end='')
    print("\n\n")

    return

def showCentroids(centroids):
    for i in range(0,len(centroids[0,:]-1)):
        plt.subplot(5, 2, i+1)
        plt.imshow(centroids[:,i].reshape(28,28))

    plt.pause(0.005)
    plt.show()
    return


alpha = 0.05

#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
centroids = np.zeros((28*28, 10))
centroidIndexs = np.zeros(10)
x_clusterInd = np.zeros(x_train.shape[0]).astype(int)


###pick which instances will be centroids

#first of each class in train is centroid
for i in range(0,10):
    j = 0
    while(y_train[j] != i):
        j = j+1
    centroids[:,i] = x_train[j].flatten()
    centroidIndexs[i] = j
    x_clusterInd[j] = -1    #-1 indicates it is a centroid, not to be used in training

    print("index: " + str(j) + "\n digit: " + str(y_train[j]))

plt.figure(1)
showCentroids(centroids)

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
    #centroids[:, x_clusterInd[i]] = centroids[:, x_clusterInd[i]] + (alpha*xi)
    centroids[:, x_clusterInd[i]] = (1-alpha) * centroids[:, x_clusterInd[i]] + (alpha * xi)

    #evaluate accuracy occasionally
    if i % 500 == 0:
        accu = accuracy_eval(i)
        print("Overall Accuracy: " + str(round(accu*100, 3)) + "%")
        accuracy_eval_perDigit(i)

        #plt.figure(2)
        #showCentroids(centroids)


plt.figure(3)
showCentroids(centroids)








raw_input('Press Enter to exit')
print("donezo")