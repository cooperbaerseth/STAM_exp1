from __future__ import print_function
import random

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


plt.interactive(True)


#the STAM class

class STAM_module:

    #class variables shared by all instances go here

    def __init__(self, field_size, num_clusts):
        self.field_size = field_size
        self.num_clusts = num_clusts

        #allocate memory for instance input and clusters
        self.input = np.zeros((self.field_size*self.field_size, 1))       #will hold the input of a STAM (AKA the image in the STAM's receptive field)
        self.centroids = np.zeros((self.field_size*self.field_size, num_clusts))        #will hold the centroids corresponding to this STAM

    def take_input(self, input):
        self.input = input





























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

    #add to accuracy graph datapoints to plot later
    accuGraph_points[progress/div - 1, :] = accu[2, :]

    #show confusion matrix
    conf_mat(progress)

    return

def show_AccuGraph():

    fig = plt.figure(1)
    ax = plt.subplot(111)

    for i in range(0, 5):
        ax.plot(np.arange(0, x_train.shape[0]-1, div), accuGraph_points[:, i], linewidth=2.0, label=str(i))
    for i in range(5, 10):
        ax.plot(np.arange(0, x_train.shape[0]-1, div), accuGraph_points[:, i], '--', linewidth=2.0, label=str(i))

    plt.xlim(xmax=x_train.shape[0]+3000)
    plt.title('Accuracy Per Class Over Time', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xlabel('Total Iterations', fontweight='bold')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show(fig)
    return

def conf_mat(n):
    confMat = np.zeros((10, 10))      #holds the values of the confusion matrix

    #populate the matrix
    for i in range(0, n):
        confMat[x_clusterInd[i], y_train[i]] = confMat[x_clusterInd[i], y_train[i]] + 1

    #show confusion matrix
    confMat_dFrame = pd.DataFrame(confMat, range(confMat.shape[0]), range(confMat.shape[1]))
    fig = plt.figure(2)
    fig.clear()
    plt.title("True Labels", fontweight='bold')
    sn.heatmap(confMat_dFrame, annot=True, fmt='g')

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

def initCents_avg(n):                   #if n = float("inf"), average all examples together
    global centroids

    if n == float("inf"):
        avgCents_full = np.zeros((28*28, 10))
        cent_count = np.zeros(10)

        #sum
        for i in range(0, x_train.shape[0]):
            avgCents_full[:, y_train[i]] = avgCents_full[:, y_train[i]] + x_train[i].flatten()
            cent_count[y_train[i]] = cent_count[y_train[i]] + 1

        #divide
        centroids = avgCents_full / cent_count[None, :]

    else:
        avgCents = np.zeros((28 * 28, 10, n))
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

def initCents_close2avg():
    global centroids

    temp_cents = np.zeros(centroids.shape)
    best_dists = np.full((10, 1), float("inf"))

    #populate centroids with averages of all instances in training set
    initCents_avg(float("inf"))

    #pick instances that are closest to global averages per class
    for i in range(0, x_train.shape[0]):
        xi = x_train[i].flatten()
        if np.linalg.norm(xi - centroids[:, y_train[i]]) < best_dists[y_train[i]]:
            temp_cents[:, y_train[i]] = xi
            best_dists[y_train[i]] = np.linalg.norm(xi - centroids[:, y_train[i]])
            centroidIndexs[y_train[i]] = i

    centroids = temp_cents

    return

alpha = 0.005
div = 1000

#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
centroids = np.zeros((28*28, 10))
centroidIndexs = np.full((10, 1), -1)
x_clusterInd = np.zeros(x_train.shape[0]).astype(int)
accuGraph_points = np.zeros((x_train.shape[0]/div, 10))       #will hold all the datapoints over time for the accuracy of each class


###INITIALIZE CENTROIDS

#first of each class in train is centroid (first come first serve)
#initCents_firstCome()

#random of each class
#initCents_pickRands()
#initCents_rands_alt()

#average of n examples
#initCents_avg(3)
initCents_avg(float("inf"))

#closest to global average
#initCents_close2avg()

plt.figure(3)
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
    if (i+1) % div == 0:

        accu = accuracy_eval(i)
        print("Overall Accuracy: " + str(round(accu*100, 3)) + "%")
        accuracy_eval_perDigit(i)

        plt.figure(4)
        showCentroids(centroids)
        raw_input('Press Enter to exit')


plt.figure(5)
showCentroids(centroids)


show_AccuGraph()





raw_input('Press Enter to exit')
print("donezo")