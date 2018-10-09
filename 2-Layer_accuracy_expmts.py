from __future__ import print_function
import random
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sn
import pandas as pd
from collections import Counter
from tempfile import TemporaryFile
from matplotlib.patches import Rectangle


plt.interactive(True)

NUM_OF_CLUSTERS = 10            #the use of this variable will change a lot as we progress, but for now it is static during the program


#The Layer Class
class Layer:

    def __init__(self, name, recField_size, stride, alpha, initCents):
        self.recField_size = recField_size
        self.stride = stride
        self.alpha = alpha
        self.name = name

        self.initCentroids = initCents      #this field should be unnecessary as we progress

        ############################################
        # Create STAMs and Initialize STAM Centroids
        ############################################

        #Create/Initialize the layer's STAMs... number of STAMs == number of receptive fields
        self.num_STAMs = int(np.power(np.floor((self.initCentroids[0].shape[0]-self.recField_size)/self.stride)+1, 2))
        self.STAMs = [[STAM(recField_size=self.recField_size, alpha=self.alpha) for j in range(int(np.sqrt(self.num_STAMs)))] for i in range(int(np.sqrt(self.num_STAMs)))]

        #Set all STAMs' initial centroid states
        self.set_initCentroids()

    def set_STAM_input(self):
        stride = self.stride
        recField_size = self.recField_size

        for i in range(0, int(np.sqrt(self.num_STAMs))):
            for j in range(0, int(np.sqrt(self.num_STAMs))):
                startI = i * stride
                endI = i * stride + recField_size
                startJ = j * stride
                endJ = j * stride + recField_size

                self.STAMs[i][j].input = self.input_image[startI:endI][:, startJ:endJ]

    def set_initCentroids(self):
        stride = self.stride
        recField_size = self.recField_size

        for i in range(0, int(np.sqrt(self.num_STAMs))):
            for j in range(0, int(np.sqrt(self.num_STAMs))):
                for c in range(NUM_OF_CLUSTERS):
                    startI = i * stride
                    endI = i * stride + recField_size
                    startJ = j * stride
                    endJ = j * stride + recField_size

                    self.STAMs[i][j].centroids[c] = self.initCentroids[c, startI:endI, startJ:endJ]

    def visualize_recFields(self, pick, cent=-1):
        STAMs = self.STAMs
        recFields = np.zeros((len(STAMs)*len(STAMs[0]), STAMs[0][0].input.shape[0], STAMs[0][0].input.shape[0]))
        border = 2
        images_amount = recFields.shape[0]
        row_amount = int(np.sqrt(images_amount))
        col_amount = int(np.sqrt(images_amount))
        image_height = recFields[0].shape[0]
        image_width = recFields[0].shape[1]

        #populate recFields matrix
        count = 0
        for i in range(0, len(STAMs)):
            for j in range(0, len(STAMs[0])):
                if pick == "input":
                    recFields[count] = STAMs[i][j].input
                elif pick == "centroid":
                    recFields[count] = STAMs[i][j].centroids[cent]
                count += 1

        all_filter_image = np.full((row_amount * image_height + border * row_amount,
                                     col_amount * image_width + border * col_amount), float('inf'))

        for filter_num in range(images_amount):
            start_row = image_height * (filter_num / col_amount) + \
                        (filter_num / col_amount + 1) * border

            end_row = start_row + image_height

            start_col = image_width * (filter_num % col_amount) + \
                        (filter_num % col_amount + 1) * border

            end_col = start_col + image_width

            all_filter_image[start_row:end_row, start_col:end_col] = \
                recFields[filter_num]

        # save image instead of showing every time
        self.recField_img = all_filter_image
        '''
        plt.figure()
        plt.imshow(all_filter_image)
        plt.title(self.name + " Receptive Fields")
        plt.axis('off')
        '''

        return

    def run_STAMs(self):
        for i in range(0, int(np.sqrt(self.num_STAMs))):
            for j in range(0, int(np.sqrt(self.num_STAMs))):
                self.STAMs[i][j].get_output()

    def append_centContrib(self, istart, iend, jstart, jend, cent):
        for i in range(istart, iend):
            for j in range(jstart, jend):
                self.pixel_centContrib[i][j].append(cent)

    def get_output(self):
        self.output_image = np.zeros(self.input_image.shape, dtype=float)
        count_image = np.zeros(self.input_image.shape)
        self.pixel_centContrib = [[[] for i in range(self.input_image.shape[0])] for j in range(self.input_image.shape[0])]  # 2D list in which each element corresponds to a pixel in the output image
                                                                                                                             #  and contains a list of the centroids that contributed to it's value
        stride = self.stride
        recField_size = self.recField_size

        # Run STAMs in layers (get STAM output)
        self.run_STAMs()

        for i in range(int(np.sqrt(self.num_STAMs))):
            for j in range(int(np.sqrt(self.num_STAMs))):
                startI = i * stride
                endI = i * stride + recField_size
                startJ = j * stride
                endJ = j * stride + recField_size

                #print("endI: " + str(endI) + "\nendJ: " + str(endJ) + "\n")

                self.output_image[startI:endI][:, startJ:endJ] += self.STAMs[i][j].output
                count_image[startI:endI][:, startJ:endJ] += 1
                self.append_centContrib(startI, endI, startJ, endJ, self.STAMs[i][j].output_cent)

        self.output_image = np.round(self.output_image / count_image)

    def feed(self, img):
        #This function takes the input for the layer and performs all of the layer's tasks

        #set the layer's input image
        self.input_image = img

        #Give each STAM its input
        self.set_STAM_input()
        #plt.figure()
        #plt.imshow(self.input_image); plt.title(self.name + " Input Image")
        self.visualize_recFields("input")

        #Create layer's output image
        self.get_output()
        #plt.figure()
        #plt.imshow(self.output_image); plt.title(self.name + " Layer Output Image")

        return

    def test_output_construction(self, img):
        # This function's purpose is to test the functionality of the layer's output image. To do this, we pass an input
        #   into the layer, and distribute it to the STAMs. We then construct the layer's output from the STAMs' INPUT,
        #   thereby excluding clustering from the process. At this point, if the layer's output image exactly matches
        #   it's input (validated by euclidean distance being 0), the output reconstruction is functioning properly.

        # set STAMs' input
        self.input_image = img
        self.num_STAMs = int(np.power(np.floor((self.input_image.shape[0] - self.recField_size) / self.stride) + 1, 2))
        self.STAMs = [[STAM(recField_size=self.recField_size, alpha=self.alpha) for j in range(int(np.sqrt(self.num_STAMs)))] for i in range(int(np.sqrt(self.num_STAMs)))]
        self.set_STAM_input()
        plt.figure()
        plt.imshow(self.input_image); plt.title(self.name + " Input Image")
        self.visualize_recFields("input")

        # reconstruct output with STAMs' input
        self.output_image = np.zeros(self.input_image.shape, dtype=float)
        count_image = np.zeros(self.input_image.shape)
        stride = self.stride
        recField_size = self.recField_size

        for i in range(0, int(np.sqrt(self.num_STAMs))):
            for j in range(0, int(np.sqrt(self.num_STAMs))):
                startI = i * stride
                endI = i * stride + recField_size
                startJ = j * stride
                endJ = j * stride + recField_size

                print("endI: " + str(endI) + "\nendJ: " + str(endJ) + "\n")

                self.output_image[startI:endI][:, startJ:endJ] += self.STAMs[i][j].input    # the key line here... changed STAM output to STAM input
                count_image[startI:endI][:, startJ:endJ] += 1
        self.output_image = np.round(self.output_image / count_image)

        # show layer output image... should be same as input
        plt.figure()
        plt.imshow(self.output_image); plt.title(self.name + " Output Image")

        # difference image
        dist = np.linalg.norm(self.input_image.flatten() - self.output_image.flatten())
        diff_img = self.input_image - self.output_image
        plt.figure()
        plt.imshow(diff_img)
        plt.title('Euclidean Distance = ' + str(dist))

        return

    def show_centroidContent(self, actual_cent, majority=False):
        # This function creates figures which attempt to make more obvious the centroids used by each STAM in the
        #   layer's output image.
        row_col = int(np.sqrt(self.num_STAMs))
        STAM_centsImg = np.zeros((row_col, row_col))
        correct_centImg = np.full(self.output_image.shape, float('inf'))


        # This image shows which centroid was used by each STAM in the layer. Each cell corresponds to an individual STAM.
        for i in range(row_col):
            for j in range(row_col):
                STAM_centsImg[i, j] = self.STAMs[i][j].output_cent

        STAM_centsImg_df = pd.DataFrame(STAM_centsImg, range(STAM_centsImg.shape[0]), range(STAM_centsImg.shape[0]))
        plt.figure()
        sn.heatmap(STAM_centsImg_df, annot=True, fmt='g')

        # This image shows the pixels in the ouput image that used the 'correct' centroid. If using 'majority',
        #   pixels are only shown if the correct centroid was used the most out of all that contributed to the
        #   pixel value.
        for i in range(len(self.pixel_centContrib)):
            for j in range(len(self.pixel_centContrib[0])):
                if majority == False:
                    if actual_cent in self.pixel_centContrib[i][j]:
                        correct_centImg[i][j] = self.output_image[i][j]
                else:
                    c = Counter(self.pixel_centContrib[i][j]).most_common()
                    for k in range(len(c)):
                        if c[k][1] == c[0][1]:
                            if c[k][0] == actual_cent:
                                correct_centImg[i][j] = self.output_image[i][j]
                                break
        plt.figure()
        plt.imshow(correct_centImg); plt.title(self.name + " Output Centroid Content")

        plt.pause(0.005)
        return

    def construct_STAMCentroids(self):
        STAM_cents = np.zeros((NUM_OF_CLUSTERS, self.input_image.shape[0], self.input_image.shape[1]), dtype=float)
        count_image = np.zeros((NUM_OF_CLUSTERS, self.input_image.shape[0], self.input_image.shape[1]))
        stride = self.stride
        recField_size = self.recField_size

        plt.figure()
        for k in range(NUM_OF_CLUSTERS):
            for i in range(int(np.sqrt(self.num_STAMs))):
                for j in range(int(np.sqrt(self.num_STAMs))):
                    startI = i * stride
                    endI = i * stride + recField_size
                    startJ = j * stride
                    endJ = j * stride + recField_size


                    STAM_cents[k][startI:endI][:, startJ:endJ] += self.STAMs[i][j].centroids[k]
                    count_image[k][startI:endI][:, startJ:endJ] += 1
            STAM_cents[k] = np.round(STAM_cents[k] / count_image[k])

            plt.subplot(5, 2, k+1)
            plt.imshow(STAM_cents[k])

        return

#The STAM Class
class STAM:

    def __init__(self, recField_size, alpha):
        self.rf_size = recField_size
        self.input = np.zeros((self.rf_size , self.rf_size ))
        self.output = np.zeros((self.rf_size , self.rf_size ))
        self.output_cent = -1
        self.centroids = np.zeros((NUM_OF_CLUSTERS, self.rf_size , self.rf_size ))
        self.alpha = alpha

    def adjust_centroid(self, ind):
        self.centroids[ind] = (1 - self.alpha) * self.centroids[ind] + (self.alpha * self.input)

    def get_output(self):
        # find closest centroid
        smallest = float("inf")  # holds distance through iterations
        close_ind = -1          # index of the closest centroid
        for i in range(NUM_OF_CLUSTERS):
            if np.linalg.norm(self.input.flatten() - self.centroids[i, :, :].flatten()) < smallest:
                smallest = np.linalg.norm(self.input.flatten() - self.centroids[i, :, :].flatten())
                close_ind = i
        self.output = self.centroids[close_ind]
        self.output_cent = close_ind
        self.adjust_centroid(close_ind)


#Init Methods for initial centroids
def initCents_avg(centroids_initial, n):                   #if n = float("inf"), average all examples together

    if n == float("inf"):
        avgCents_full = np.zeros((NUM_OF_CLUSTERS, 28*28))
        cent_count = np.zeros(NUM_OF_CLUSTERS)

        #sum
        for i in range(0, x_train.shape[0]):
            avgCents_full[y_train[i], :] = avgCents_full[y_train[i], :] + x_train[i].flatten()
            cent_count[y_train[i]] = cent_count[y_train[i]] + 1

        #divide
        centroids_initial = (avgCents_full / cent_count[:, None]).reshape((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))

    else:
        avgCents = np.zeros((n, 10, 28 * 28))
        avgCents_ind = np.full((n, 10), -1)

        #fill the matrix in order to average
        for i in range(0, n):
            filled = 0
            while filled != 10:
                pick = random.randint(0, y_train.size-1)
                if avgCents_ind[i, y_train[pick]] == -1 and pick not in avgCents_ind:
                    avgCents[i, y_train[pick], :] = x_train[pick].flatten()
                    avgCents_ind[i, y_train[pick]] = pick
                    filled = filled + 1

        #average by 3rd dimension
        temp = avgCents.mean(axis=0)
        for i in range(0, 10):
            centroids_initial[i, :, :] = (temp[i, :]).reshape((x_train[0].shape[0], x_train[0].shape[0]))
    return centroids_initial

def initCents_close2avg(centroids_initial):

    temp_cents = np.zeros((centroids_initial.shape[0], centroids_initial.shape[1] * centroids_initial.shape[1]))
    best_dists = np.full((NUM_OF_CLUSTERS, 1), float("inf"))

    #populate centroids with averages of all instances in training set
    centroids_initial = initCents_avg(centroids_initial, float("inf"))

    #pick instances that are closest to global averages per class
    for i in range(0, x_train.shape[0]):
        xi = x_train[i].flatten()
        if np.linalg.norm(xi - centroids_initial[y_train[i], :, :].flatten()) < best_dists[y_train[i]]:
            temp_cents[y_train[i], :] = xi
            best_dists[y_train[i]] = np.linalg.norm(xi - centroids_initial[y_train[i], :, :].flatten())
            #centroidIndexs[y_train[i]] = i

    centroids_initial = temp_cents.reshape((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))

    return centroids_initial

def initCents_pickRands(centroids):
    n = 1
    cent_picks = np.zeros((10, n))  #will hold the random pick'th instance in sample
    tr_stat = plt.hist(y_train)     #get number of instances in each class
    plt.close('all')

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
        centroids[i] = x_train[j-1]

    return centroids

def shuffleHist(x, x_shuff):
    # Show original image and shuffled have same values contained
    fig, subp = plt.subplots(2, 2)
    subp[0, 0].hist(x.flatten())
    subp[0, 1].hist(x_shuff)
    subp[1, 0].imshow(x)
    subp[1, 1].imshow(x_shuff.reshape(x.shape))
    plt.suptitle('Original vs Shuffled Histograms')
    plt.show()

def initCents_shuff(centroids_initial):
    temp_cents = np.zeros((NUM_OF_CLUSTERS, centroids_initial.shape[1]*centroids_initial.shape[1]))
    for i in range(NUM_OF_CLUSTERS):
        temp_cents[i] = centroids_initial[i].flatten()
        np.random.shuffle(temp_cents[i])
        #shuffleHist(centroids_initial[i], temp_cents[i])       # shows histogram equivilance
        centroids_initial[i] = temp_cents[i].reshape(centroids_initial[0].shape)

    # Take out 1 since it was the centroid with the smallest sum
    # centroids_initial[1] = temp_cents[0].reshape(centroids_initial[0].shape)
    # for i in range(NUM_OF_CLUSTERS):
    #     print(np.sum(centroids_initial[i]))

    return centroids_initial

def showCentroids(centroids):
    plt.figure()
    for i in range(NUM_OF_CLUSTERS):
        plt.subplot(5, 2, i+1)
        plt.imshow(centroids[i, :])

    plt.pause(0.005)
    plt.show()
    return

def feed_centroid(centroids_initial, layer, cent=0):
    cent = cent
    layer.feed(centroids_initial[cent, :, :])
    dist = np.linalg.norm(centroids_initial[cent, :, :].flatten() - layer.output_image.flatten())
    diff_img = centroids_initial[cent, :, :] - layer.output_image
    plt.figure()
    plt.imshow(diff_img)
    plt.title('Euclidean Distance = ' + str(dist))

    return

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

'''
**********************
*********MAIN*********
**********************
'''

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize global centroids
centroids_init_c2avg = np.zeros((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))
centroids_init_c2avg = initCents_close2avg(centroids_init_c2avg)

centroids_init_rand1 = np.zeros((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))
centroids_init_rand1 = initCents_pickRands(centroids_init_rand1)


################
################
# Supervised
################
################

# Layered Models
# alphas = [0.05, 0.025, 0.005]
# for a in range(len(alphas)):
#     conf_mat = np.zeros((NUM_OF_CLUSTERS, NUM_OF_CLUSTERS))  # True class will be rows, classified will be cols
#     L1 = Layer("L1", 7, 7, alphas[a], centroids_init_c2avg)
#     L_final = Layer("L Final", 28, 28, alphas[a], centroids_init_c2avg)
#
#     # Feed
#     div = 100
#     accu_dps = np.zeros((1, x_train.shape[0] / div));
#     a_count = 0
#     for i in range(x_train.shape[0]):
#         L1.feed(x_train[i])
#         L_final.feed(L1.output_image)
#         classified = L_final.STAMs[0][0].output_cent
#
#         conf_mat[y_train[i]][classified] += 1
#
#         print(str(round((i / float(x_train.shape[0])) * 100, 1)) + "% complete...")
#
#         if (i + 1) % div == 0:
#             # Save accuracy datapoint
#             accu_dps[0][a_count] = round((sum(conf_mat.diagonal()) / float(i + 1)) * 100, 2)
#             a_count += 1
#
#             # Show progress after some iterations
#             # plt.close('all')
#             # accu = round((sum(conf_mat.diagonal()) / float(i+1)) * 100, 2)
#             # sn.heatmap(conf_mat, annot=True, fmt='g'); plt.title(str(accu) + "% Accuracy (using " + str(i + 1) + " examples)")
#             # showCentroids(L_single.STAMs[0][0].centroids); plt.suptitle("L Single Centroids")
#             # showCentroids(centroids_initial); plt.suptitle("Initial Centroids")
#             # plt.pause(0.005)
#             # raw_input('Press Enter to exit')
#
#         # Show relavent figures after one iteration
#         # print(L_final.STAMs[0][0].output_cent)
#         # plt.figure(); plt.imshow(L1.input_image); plt.title("L1 Input Image")
#         # plt.figure(); plt.imshow(L_final.input_image); plt.title("L Final Input Image")
#         # plt.figure(); plt.imshow(L_final.output_image); plt.title("L Final Output Image")
#         # showCentroids(centroids_initial)
#         # plt.pause(0.005)
#         # plt.show()
#         # raw_input('Press Enter to exit')
#         # plt.close('all')
#
#     plt.close('all')
#     accu = round((sum(conf_mat.diagonal()) / float(x_train.shape[0])) * 100, 2)
#     plt.figure()
#     x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0] / div)
#     plt.plot(x_plot, accu_dps[0]);
#     plt.title("Classification Accuracy After X Examples")
#
#     plt.figure()
#     sn.heatmap(conf_mat, annot=True, fmt='g');
#     plt.title(str(accu) + "% Accuracy (using " + str(i + 1) + " examples)")
#
#     showCentroids(L_final.STAMs[0][0].centroids);
#     plt.suptitle("L Final Centroids")
#     showCentroids(centroids_init_c2avg);
#     plt.suptitle("Initial Centroids")
#     # print("Final Layered Accuracy: " + str(layered_score / float(x_train.shape[0])))
#
#     # Save PDF
#     multipage("AlphaLayered_" + str(a))


# Single STAM Models
#
# alphas = [0.05, 0.025, 0.005]
# for a in range(len(alphas)):
#     conf_mat = np.zeros((NUM_OF_CLUSTERS, NUM_OF_CLUSTERS))  # True class will be rows, classified will be cols
#     L_single = Layer("L Single", 28, 28, alphas[a], centroids_init_c2avg)
#
#     # Feed
#     div = 100
#     accu_dps = np.zeros((1, x_train.shape[0]/div)); a_count = 0
#     for i in range(x_train.shape[0]):
#         L_single.feed(x_train[i])
#         classified = L_single.STAMs[0][0].output_cent
#
#         conf_mat[y_train[i]][classified] += 1
#
#         print(str(round((i/float(x_train.shape[0]))*100, 1)) + "% complete...")
#
#         if (i + 1) % div == 0:
#             # Save accuracy datapoint
#             accu_dps[0][a_count] = round((sum(conf_mat.diagonal()) / float(i+1)) * 100, 2)
#             a_count += 1
#
#             # Show progress after some iterations
#             # plt.close('all')
#             # accu = round((sum(conf_mat.diagonal()) / float(i+1)) * 100, 2)
#             # sn.heatmap(conf_mat, annot=True, fmt='g'); plt.title(str(accu) + "% Accuracy (using " + str(i + 1) + " examples)")
#             # showCentroids(L_single.STAMs[0][0].centroids); plt.suptitle("L Single Centroids")
#             # showCentroids(centroids_initial); plt.suptitle("Initial Centroids")
#             # plt.pause(0.005)
#             # raw_input('Press Enter to exit')
#
#         # Show relavent figures after one iteration
#         # print(L_final.STAMs[0][0].output_cent)
#         # plt.figure(); plt.imshow(L1.input_image); plt.title("L1 Input Image")
#         # plt.figure(); plt.imshow(L_final.input_image); plt.title("L Final Input Image")
#         # plt.figure(); plt.imshow(L_final.output_image); plt.title("L Final Output Image")
#         # showCentroids(centroids_initial)
#         # plt.pause(0.005)
#         # plt.show()
#         # raw_input('Press Enter to exit')
#         # plt.close('all')
#
#
#     plt.close('all')
#     accu = round((sum(conf_mat.diagonal()) / float(x_train.shape[0])) * 100, 2)
#     plt.figure()
#     x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
#     plt.plot(x_plot, accu_dps[0]); plt.title("Classification Accuracy After X Examples")
#
#     plt.figure()
#     sn.heatmap(conf_mat, annot=True, fmt='g'); plt.title(str(accu) + "% Accuracy (using " + str(i+1) + " examples)")
#
#     showCentroids(L_single.STAMs[0][0].centroids); plt.suptitle("L Single Centroids")
#     showCentroids(centroids_initial); plt.suptitle("Initial Centroids")
#     #print("Final Layered Accuracy: " + str(layered_score / float(x_train.shape[0])))
#
#     # Save PDF
#     #multipage("AlphaSingle_" + str(a))

def getAccuracies(L1, L_final, L_single, div=100):
    conf_mat = np.zeros((2, NUM_OF_CLUSTERS, NUM_OF_CLUSTERS))  # True class will be rows, classified will be cols
    accu_dps = np.zeros((2, NUM_OF_CLUSTERS+1, x_train.shape[0] / div));
    a_count = 0
    classified = np.zeros(2, int)
    for i in range(x_train.shape[0]):
        # Feed layered model
        L1.feed(x_train[i])
        L_final.feed(L1.output_image)

        # Feed single STAM model
        L_single.feed(x_train[i])

        classified[0] = L_final.STAMs[0][0].output_cent
        classified[1] = L_single.STAMs[0][0].output_cent

        conf_mat[0][y_train[i]][classified[0]] += 1
        conf_mat[1][y_train[i]][classified[1]] += 1

        print(str(round((i / float(x_train.shape[0])) * 100, 1)) + "% complete...")

        if (i + 1) % div == 0:
            # Save accuracy datapoints
            # Total Accuracy
            accu_dps[0][0][a_count] = round((sum(conf_mat[0].diagonal()) / float(i + 1)) * 100, 2)
            accu_dps[1][0][a_count] = round((sum(conf_mat[1].diagonal()) / float(i + 1)) * 100, 2)
            # Per Cluster Accuracy
            for j in range(NUM_OF_CLUSTERS):
                accu_dps[0][j+1][a_count] = round((conf_mat[0][j][j] / sum(conf_mat[0][j])) * 100, 2)
                accu_dps[1][j+1][a_count] = round((conf_mat[1][j][j] / sum(conf_mat[1][j])) * 100, 2)
            a_count += 1

        # Show progress with figures
        # if (i+1)%30000 == 0:
        #     plt.close('all')
        #     accu = round((sum(conf_mat[0].diagonal()) / float(i+1)) * 100, 2)
        #     sn.heatmap(conf_mat[0], annot=True, fmt='g'); plt.title(str(accu) + "% Layered Accuracy (using " + str(i + 1) + " examples)")
        #     showCentroids(L_final.STAMs[0][0].centroids); plt.suptitle("Layered Centroids")
        #     accu = round((sum(conf_mat[1].diagonal()) / float(i+1)) * 100, 2)
        #     plt.figure()
        #     sn.heatmap(conf_mat[1], annot=True, fmt='g'); plt.title(str(accu) + "% Single STAM Accuracy (using " + str(i + 1) + " examples)")
        #     showCentroids(L_single.STAMs[0][0].centroids); plt.suptitle("Single STAM Centroids")
        #     plt.pause(0.005)
        #     raw_input('Press Enter to exit')

    return accu_dps

# div = 100
#
# # Close2Avg Models
# L1 = Layer("L1", 7, 7, 0.005, centroids_init_c2avg)
# L_final = Layer("L Final", 28, 28, 0.005, centroids_init_c2avg)
# L_single = Layer("L Single", 28, 28, 0.005, centroids_init_c2avg)
# accu_c2avg = getAccuracies(L1, L_final, L_single, div)
#
# # Random 1 Models
# L1 = Layer("L1", 7, 7, 0.005, centroids_init_rand1)
# L_final = Layer("L Final", 28, 28, 0.005, centroids_init_rand1)
# L_single = Layer("L Single", 28, 28, 0.005, centroids_init_rand1)
# accu_rand1 = getAccuracies(L1, L_final, L_single, div)

# Plot Results

# Model Accuracy Comp
# plt.figure()
# x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
# plt.plot(x_plot, accu_c2avg[0][0][:], label='Layered')
# plt.plot(x_plot, accu_c2avg[1][0][:], label='Single STAM')
# plt.title("Classification Accuracy (Close2Avg Init Method)"); plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(x_plot, accu_rand1[0][0][:], label='Layered')
# plt.plot(x_plot, accu_rand1[1][0][:], label='Single STAM')
# plt.title("Classification Accuracy (Random Pick Init Method)"); plt.legend()
# plt.show()

# Per Cluster Accuracy Clos2Avg
# plt.figure()
# ax = plt.subplot(111)
# x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
# ax.plot(x_plot, accu_c2avg[0][0][:], label='Overall')
# for i in range(NUM_OF_CLUSTERS):
#     plt.plot(x_plot, accu_c2avg[0][i][:], label='Clust ' + str(i))
# plt.title("Layered Classification Accuracy (Close2Avg Init Method)");
# box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()
#
# plt.figure()
# ax = plt.subplot(111)
# x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
# ax.plot(x_plot, accu_c2avg[1][0][:], label='Overall')
# for i in range(NUM_OF_CLUSTERS):
#     plt.plot(x_plot, accu_c2avg[1][i][:], label='Clust ' + str(i))
# plt.title("Single STAM Classification Accuracy (Close2Avg Init Method)");
# box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()


def getResults_alpha(div=100):
    alphas = [0.05, 0.025, 0.005]
    accu_dps = np.zeros((2, len(alphas), x_train.shape[0] / div));
    for a in range(len(alphas)):
        L1 = Layer("L1", 7, 7, alphas[a], centroids_init_c2avg)
        L_final = Layer("L Final", 28, 28, alphas[a], centroids_init_c2avg)
        L_single = Layer("L Single", 28, 28, alphas[a], centroids_init_c2avg)
        conf_mat = np.zeros((2, NUM_OF_CLUSTERS, NUM_OF_CLUSTERS), int)
        a_count = 0
        classified = np.zeros(2, int)
        for i in range(x_train.shape[0]):
            # Feed layered model
            L1.feed(x_train[i])
            L_final.feed(L1.output_image)

            # Feed single STAM model
            L_single.feed(x_train[i])

            classified[0] = L_final.STAMs[0][0].output_cent
            classified[1] = L_single.STAMs[0][0].output_cent

            conf_mat[0][y_train[i]][classified[0]] += 1
            conf_mat[1][y_train[i]][classified[1]] += 1

            print(str(round((i / float(x_train.shape[0])) * 100, 1)) + "% complete...")

            if (i + 1) % div == 0:
                # Save accuracy datapoint
                accu_dps[0][a][a_count] = round((sum(conf_mat[0].diagonal()) / float(i + 1)) * 100, 2)
                accu_dps[1][a][a_count] = round((sum(conf_mat[1].diagonal()) / float(i + 1)) * 100, 2)
                a_count += 1
    return accu_dps



# div = 100
# accu_alphaComp = getResults_alpha(div)
# np.save('alpha_comp.npy', accu_alphaComp)
#
# plt.figure()
# x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
# plt.plot(x_plot, accu_alphaComp[0][0][:], label='Alpha = 0.05')
# plt.plot(x_plot, accu_alphaComp[0][1][:], label='Alpha = 0.025')
# plt.plot(x_plot, accu_alphaComp[0][2][:], label='Alpha = 0.005')
# plt.title("Layered Classification Accuracy After X Examples"); plt.legend()
# plt.show()
# plt.figure()
# x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
# plt.plot(x_plot, accu_alphaComp[1][0][:], label='Alpha = 0.05')
# plt.plot(x_plot, accu_alphaComp[1][1][:], label='Alpha = 0.025')
# plt.plot(x_plot, accu_alphaComp[1][2][:], label='Alpha = 0.005')
# plt.title("Single STAM Classification Accuracy After X Examples"); plt.legend()
# plt.show()


################
################
# Unsupervised
################
################

# Initialize centroids with shuffled Close2Avg cents
centroids_init_shuff = np.zeros((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))
centroids_init_shuff = initCents_shuff(initCents_close2avg(centroids_init_shuff))
#showCentroids(centroids_init_shuff)

def heatmap_majClust(mat):
    # Each column in mat corresponds to a cluster, and each row in the column corresponds to a class of object.
    # For each cluster, highlight the majority class by putting a rectangle around the majority class' cell.
    ax = sn.heatmap(mat, annot=True, fmt='g')
    for i in range(NUM_OF_CLUSTERS):
        if np.max(mat[:, i]) != 0:
            ax.add_patch(Rectangle((i, np.argmax(mat[:, i])), 1, 1, fill=False, edgecolor='blue', lw=3))
    plt.show()

def getHomogeneity(L1, L_final, L_single, div=100):
    conf_mat = np.zeros((2, NUM_OF_CLUSTERS, NUM_OF_CLUSTERS))  # True class will be rows, clustered will be cols
    homog_dps = np.zeros((2, NUM_OF_CLUSTERS+1, x_train.shape[0] / div));
    a_count = 0
    clustered = np.zeros(2, int)
    for i in range(x_train.shape[0]):
        # Feed layered model
        L1.feed(x_train[i])
        L_final.feed(L1.output_image)

        # Feed single STAM model
        L_single.feed(x_train[i])

        clustered[0] = L_final.STAMs[0][0].output_cent
        clustered[1] = L_single.STAMs[0][0].output_cent

        conf_mat[0][y_train[i]][clustered[0]] += 1
        conf_mat[1][y_train[i]][clustered[1]] += 1

        print(str(round((i / float(x_train.shape[0])) * 100, 1)) + "% complete...")

        if (i + 1) % div == 0:
            # Save homogeneity datapoints
            # Per Cluster Homogeneity
            for j in range(NUM_OF_CLUSTERS):
                if np.max(conf_mat[0][:, j]) != 0:
                    homog_dps[0][j][a_count] = round((np.max(conf_mat[0][:, j]) / np.sum(conf_mat[0][:, j])) * 100, 2)
                else:
                    homog_dps[0][j][a_count] = 0

                if np.max(conf_mat[1][:, j]) != 0:
                    homog_dps[1][j][a_count] = round((np.max(conf_mat[1][:, j]) / np.sum(conf_mat[1][:, j])) * 100, 2)
                else:
                    homog_dps[1][j][a_count] = 0
            homog_dps[0][0][a_count] = np.average(homog_dps[0][1:NUM_OF_CLUSTERS-1])
            homog_dps[1][0][a_count] = np.average(homog_dps[1][1:NUM_OF_CLUSTERS-1])
            a_count += 1

        # Show progress with figures
        if (i+1)%30001 == 0:
            plt.close('all')
            heatmap_majClust(conf_mat[0]); plt.title(str(homog_dps[0][0][a_count]) + "% Layered Average Homogeneity (using " + str(i + 1) + " examples)")
            showCentroids(L_final.STAMs[0][0].centroids); plt.suptitle("Layered Centroids")
            plt.figure()
            heatmap_majClust(conf_mat[1]); plt.title(str(homog_dps[1][0][a_count]) + "% Single STAM Average Homogeneity (using " + str(i + 1) + " examples)")
            showCentroids(L_single.STAMs[0][0].centroids); plt.suptitle("Single STAM Centroids")
            plt.pause(0.005)
            raw_input('Press Enter to exit')
    return homog_dps, conf_mat


div = 100
L1 = Layer("L1", 7, 7, 0.005, centroids_init_shuff)
L_final = Layer("L Final", 28, 28, 0.005, centroids_init_shuff)
L_single = Layer("L Single", 28, 28, 0.005, centroids_init_shuff)
homogen, conf_mat = getHomogeneity(L1, L_final, L_single, div)

# Per Cluster Homogeneity Plots
plt.figure()
ax = plt.subplot(111)
x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
ax.plot(x_plot, homogen[0][0][:], label='Average')
for i in range(NUM_OF_CLUSTERS):
    plt.plot(x_plot, homogen[0][i][:], label='Clust ' + str(i))
plt.title("Layered Per-Cluster Homogeneity");
box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure()
heatmap_majClust(conf_mat[0]); plt.title(str(conf_mat[0][0][-1]) + "% Layered Average Homogeneity (using " + str(x_train.shape[0]) + " examples)")
plt.show()

plt.figure()
ax = plt.subplot(111)
x_plot = np.linspace(div, x_train.shape[0], x_train.shape[0]/div)
ax.plot(x_plot, homogen[1][0][:], label='Average')
for i in range(NUM_OF_CLUSTERS):
    plt.plot(x_plot, homogen[1][i][:], label='Clust ' + str(i))
plt.title("Single STAM Per-Cluster Homogeneity");
box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure()
heatmap_majClust(conf_mat[1]); plt.title(str(conf_mat[1][0][-1]) + "% Single STAM Average Homogeneity (using " + str(x_train.shape[0]) + " examples)")
plt.show()


plt.pause(0.005)
raw_input('Press Enter to exit')