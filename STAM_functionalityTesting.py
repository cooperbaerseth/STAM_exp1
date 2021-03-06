from __future__ import print_function
import random
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from collections import Counter


plt.interactive(True)

NUM_OF_CLUSTERS = 10            #the use of this variable will change a lot as we progress, but for now it is static during the program


#The Layer Class
class Layer:

    def __init__(self, recField_size, stride, alpha, initCents):
        self.recField_size = recField_size
        self.stride = stride
        self.alpha = alpha

        self.initCentroids = initCents      #this field should be unnecessary as we progress

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

        plt.figure()
        plt.imshow(all_filter_image)
        plt.axis('off')

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

                print("endI: " + str(endI) + "\nendJ: " + str(endJ) + "\n")

                self.output_image[startI:endI][:, startJ:endJ] += self.STAMs[i][j].output
                count_image[startI:endI][:, startJ:endJ] += 1
                self.append_centContrib(startI, endI, startJ, endJ, self.STAMs[i][j].output_cent)

        self.output_image = np.round(self.output_image / count_image)

    def feed(self, img):
        #This function takes the input for the layer and performs all of the layer's tasks

        #set the layer's input image
        self.input_image = img

        self.num_STAMs = int(np.power(np.floor((self.input_image.shape[0]-self.recField_size)/self.stride)+1, 2))

        #Create/Initialize the layer's STAMs... number of STAMs == number of receptive fields
        self.STAMs = [[STAM(recField_size=self.recField_size, alpha=self.alpha) for j in range(int(np.sqrt(self.num_STAMs)))] for i in range(int(np.sqrt(self.num_STAMs)))]

        #Set all STAMs' initial centroid states
        self.set_initCentroids()
        #self.visualize_recFields("centroid", 7)        #demonstration of a centroid visualized

        #Give each STAM its input
        self.set_STAM_input()
        plt.figure()
        plt.imshow(self.input_image)
        self.visualize_recFields("input")

        #Create layer's output image
        self.get_output()
        plt.figure()
        plt.imshow(self.output_image)

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
        plt.imshow(self.input_image)
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
        plt.imshow(self.output_image)

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
        plt.imshow(correct_centImg)

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

                    print("endI: " + str(endI) + "\nendJ: " + str(endJ) + "\n")

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
def initCents_avg(n):                   #if n = float("inf"), average all examples together
    global centroids_initial

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
    return

def initCents_close2avg():
    global centroids_initial

    temp_cents = np.zeros((centroids_initial.shape[0], centroids_initial.shape[1] * centroids_initial.shape[1]))
    best_dists = np.full((NUM_OF_CLUSTERS, 1), float("inf"))

    #populate centroids with averages of all instances in training set
    initCents_avg(float("inf"))

    #pick instances that are closest to global averages per class
    for i in range(0, x_train.shape[0]):
        xi = x_train[i].flatten()
        if np.linalg.norm(xi - centroids_initial[y_train[i], :, :].flatten()) < best_dists[y_train[i]]:
            temp_cents[y_train[i], :] = xi
            best_dists[y_train[i]] = np.linalg.norm(xi - centroids_initial[y_train[i], :, :].flatten())
            #centroidIndexs[y_train[i]] = i

    centroids_initial = temp_cents.reshape((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))

    return

def showCentroids(centroids):
    plt.figure()
    for i in range(NUM_OF_CLUSTERS):
        plt.subplot(5, 2, i+1)
        plt.imshow(centroids[i, :])

    plt.pause(0.005)
    plt.show()
    return

def feed_centroid(layer, cent=0):
    cent = cent
    layer.feed(centroids_initial[cent, :, :])
    dist = np.linalg.norm(centroids_initial[cent, :, :].flatten() - layer.output_image.flatten())
    diff_img = centroids_initial[cent, :, :] - layer.output_image
    plt.figure()
    plt.imshow(diff_img)
    plt.title('Euclidean Distance = ' + str(dist))

    return

'''
**********************
*********MAIN*********
**********************
'''

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize global centroids
centroids_initial = np.zeros((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))
#initCents_avg(float("inf"))
#initCents_avg(5)
initCents_close2avg()
#showCentroids(centroids_initial)


# Test Layer output reconstruction
#L1 = Layer(4, 2, 0.005, centroids_initial)
#L1.test_output_construction(x_train[1])

# Test if centroids will be output as themselves
#L1 = Layer(4, 2, 0.005, centroids_initial)
#feed_centroid(L1, 8)

# Show content of layer's output image with respect to STAM centroids
L1 = Layer(4, 2, 0.005, centroids_initial)
#L1 = Layer(7, 7, 0.05, centroids_initial)
m = True
for i in range(x_train.shape[0]):
    L1.feed(x_train[i])
    L1.show_centroidContent(y_train[i], m)
    showCentroids(centroids_initial)
    raw_input('Press Enter to exit')
    plt.close('all')

# Show how STAM centroids are changing after each new instance
# L1 = Layer(4, 2, 0.005, centroids_initial)
# for i in range(x_train.shape[0]):
#     L1.feed(x_train[i])
#     L1.construct_STAMCentroids()
#     plt.pause(0.005)
#     plt.show()
#     raw_input('Press Enter to exit')
#     plt.close('all')


plt.pause(0.005)
plt.show()
print('done')
raw_input('Press Enter to exit')