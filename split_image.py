from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

plt.interactive(True)
#plt.ion()

def get_recFields(input, rec_size, stride):
    num_recFields = int(np.power(np.floor((input.shape[0]-rec_size)/stride)+1, 2))
    outputs = np.zeros((num_recFields, rec_size, rec_size))

    outCount = 0
    for i in range(0, int(np.floor((input.shape[0]-rec_size)/stride)+1)):
        for j in range(0, int(np.floor((input.shape[0]-rec_size)/stride)+1)):
            startI = i*stride
            endI = i*stride + rec_size
            startJ = j*stride
            endJ = j*stride + rec_size


            outputs[outCount] = input[startI:endI][:, startJ:endJ]
            outCount = outCount + 1

    visualize_recFields(outputs)

    return outputs


def visualize_recFields(recFields):
    border = 2
    images_amount = recFields.shape[0]
    row_amount = int(np.sqrt(images_amount))
    col_amount = int(np.sqrt(images_amount))
    image_height = recFields[0].shape[0]
    image_width = recFields[0].shape[1]

    all_filter_image = np.zeros((row_amount * image_height + border * row_amount,
                                 col_amount * image_width + border * col_amount))

    for filter_num in range(images_amount):
        start_row = image_height * (filter_num / col_amount) + \
                    (filter_num / col_amount + 1) * border

        end_row = start_row + image_height

        start_col = image_width * (filter_num % col_amount) + \
                    (filter_num % col_amount + 1) * border

        end_col = start_col + image_width

        all_filter_image[start_row:end_row, start_col:end_col] = \
            recFields[filter_num]

    plt.figure(2)
    plt.imshow(all_filter_image)
    plt.axis('off')


    return


#load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(0, x_train.shape[0]):
    plt.figure(1)
    plt.imshow(x_train[i])
    get_recFields(x_train[i], 5, 1)
    plt.pause(0.005)
    plt.show()
    raw_input('Press Enter to exit')


print("Done")