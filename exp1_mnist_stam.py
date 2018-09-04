from __future__ import print_function
import random

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

def initCents_firstCome():
    for i in range(0, 10):
        j = 0
        while (y_train[j] != i):
            j = j + 1
        centroids[:, i] = x_train[j].flatten()
        centroidIndexs[i] = j
        x_clusterInd[j] = -1  # -1 indicates it is a centroid, not to be used in training

        print("index: " + str(j) + "\n digit: " + str(y_train[j]))

    return

def showCentroids(centroids):
    for i in range(0,len(centroids[0,:]-1)):
        plt.subplot(5, 2, i+1)
        plt.imshow(centroids[:,i].reshape(28,28))

    plt.pause(0.005)
    plt.show()
    return

def initCents_pickRands():
    n = 1
    cent_picks = np.zeros((10, n))  #will hold the random pick'th instance in sample
    tr_stat = plt.hist(y_train)     #get number of instances in each class

    #get indicies for random selection from train
    for i in range(0, 10):
        pick = random.randint(1, int(tr_stat[0][i]))
        while pick in cent_picks:
            pick = random.randint(1, int(tr_stat[0][i]))
        cent_picks[i, n-1] = pick

    #populate centroids
    for i in range(0, 10):
        count = 0
        j = 0
        while count != cent_picks[i, n-1]:
            if y_train[j] == i:
                count = count + 1
            j = j + 1
        centroids[:, i] = x_train[j-1].flatten()
        centroidIndexs[i] = j-1

    return

def initCents_rands_alt():
    filled = 0

    while filled != centroidIndexs.size:
        pick = random.randint(0, y_train.size-1)
        if centroidIndexs[y_train[pick]] == -1 and pick not in centroidIndexs:
            centroidIndexs[y_train[pick]] = pick
            centroids[:, y_train[pick]] = x_train[pick].flatten()
            filled = filled + 1
    return

def initCents_avg(n):
    avgCents = np.zeros((28*28, 10, n))
    avgCents_ind = np.full((10, n), -1)

    #fill the matrix in order to average
    for i in range(0, n):
        filled = 0
        while filled != 10:
            pick = random.randint(0, y_train.size-1)
            if avgCents_ind[y_train[pick], i] == -1 and pick not in avgCents_ind:
                avgCents[:, y_train[pick], i] = x_train[pick].flatten()
                avgCents_ind[y_train[pick], i] = pick
                filled = filled + 1

    #average by 3rd dimension
    temp = avgCents.mean(axis=2)
    for i in range(0,10):
        centroids[:, i] = temp[:, i]

    return

alpha = 0.05

#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
centroids = np.zeros((28*28, 10))
centroidIndexs = np.full((10, 1), -1)
x_clusterInd = np.zeros(x_train.shape[0]).astype(int)


###INITIALIZE CENTROIDS

#first of each class in train is centroid (first come first serve)
#initCents_firstCome()

#random of each class
#initCents_pickRands()
#initCents_rands_alt()

#average of n examples
initCents_avg(3)

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
    if i % 500 == 0 and i != 0:
        accu = accuracy_eval(i)
        print("Overall Accuracy: " + str(round(accu*100, 3)) + "%")
        accuracy_eval_perDigit(i)

        plt.figure(2)
        showCentroids(centroids)
        #raw_input('Press Enter to exit')


plt.figure(3)
showCentroids(centroids)








raw_input('Press Enter to exit')
print("donezo")