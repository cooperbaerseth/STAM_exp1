from STAM_classRepo import *

import matplotlib.gridspec as gridspec


def rf_split(layer, im):
    STAMs = layer.STAMs
    recFields = np.zeros((len(STAMs) * len(STAMs[0]), STAMs[0][0].input.shape[0], STAMs[0][0].input.shape[0]))
    rf = layer.recField_size
    border = 1
    images_amount = recFields.shape[0]
    row_amount = int(np.sqrt(images_amount))
    col_amount = int(np.sqrt(images_amount))
    image_height = recFields[0].shape[0]
    image_width = recFields[0].shape[1]

    # populate recFields matrix
    count = 0
    for i in range(0, len(STAMs)):
        for j in range(0, len(STAMs)):
            iStart = i * rf
            iEnd = i * rf + rf
            jStart = j * rf
            jEnd = j * rf + rf
            recFields[count] = im[iStart:iEnd][:, jStart:jEnd]
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

    plt.imshow(all_filter_image)

    return all_filter_image

def visualize(l1, l2, x, y, c2y, y_fb, c2y_fb, fb_ind):
    # Create subplot to visualize progression
    fig = plt.figure()
    gridspec.GridSpec(2, 4)

    plt.subplot2grid((2, 4), (0, 0))
    plt.imshow(x); plt.title("X"); plt.axis('off')

    plt.subplot2grid((2, 4), (0, 1))
    rf_split(l1, x); plt.title("X"); plt.axis('off')

    plt.subplot2grid((2, 4), (0, 2))
    rf_split(l1, y); plt.title("Y"); plt.axis('off')

    plt.subplot2grid((2, 4), (0, 3))
    plt.imshow(c2y); plt.title("c_2(Y)"); plt.axis('off')

    plt.subplot2grid((2, 4), (1, 0), colspan=1, rowspan=1)
    l1.showConvergenceImage(get=True); plt.title("Y Convergence"); plt.axis('off')

    plt.subplot2grid((2, 4), (1, 1), colspan=1, rowspan=1)
    rf_split(l1, l1.get_XPrime()); plt.title("X'"); plt.axis('off')

    plt.subplot2grid((2, 4), (1, 2))
    rf_split(l1, y_fb); plt.title("Y'"); plt.axis('off')

    plt.subplot2grid((2, 4), (1, 3))
    plt.imshow(c2y_fb); plt.title("c_2(Y')"); plt.axis('off')

    fig.tight_layout()
    fig.suptitle('Feedback Iteration ' + str(fb_ind), size='x-large', weight='bold')


    plt.pause(0.005)
    raw_input('Press Enter to exit')

def feedback(x, n, L1, L2):
    # x: original input image
    # n: number of feedback iterations

    L1.feed(x); X = L1.input_image
    L2.feed(L1.output_image); Y = L2.input_image; c_2Y = L2.output_image
    for i in range(n):
        L1.feed(L2.output_image, feedback=True); Y_fb = L1.output_image
        if L1.converged():
            visualize(L1, L2, X, Y, c_2Y, Y_fb, c_2Y_fb, i+1)
            plt.close('all')
            L1.unlock()
            L2.unlock()
            return
        L2.feed(L1.output_image, feedback=True); c_2Y_fb = L2.output_image
        visualize(L1, L2, X, Y, c_2Y, Y_fb, c_2Y_fb, i+1)
    return

'''
**********************
*********MAIN*********
**********************
'''

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize centroids
centroids_init_c2avg = np.zeros((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))
centroids_init_c2avg = initCents_close2avg(centroids_init_c2avg, x_train, y_train)

# Initialize Layers
l1 = Layer("L1", 7, 7, 0.005, centroids_init_c2avg)
l2 = Layer("L2", 28, 28, 0.005, centroids_init_c2avg)

for i in range(x_train.shape[0]):
    feedback(x_train[i], 10, l1, l2)



