from STAM_classRepo import *

import matplotlib.gridspec as gridspec


def visualize(l1, l2, x, y, c2y, y_fb, c2y_fb):
    # Create subplot to visualize progression
    fig = plt.figure()
    gridspec.GridSpec(4, 2)

    plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=1)
    plt.imshow(x); plt.title("X")

    plt.subplot2grid((4, 2), (1, 0))
    plt.imshow(y); plt.title("Y")

    plt.subplot2grid((4, 2), (1, 1))
    plt.imshow(c2y); plt.title("c_2(Y)")

    plt.subplot2grid((4, 2), (2, 0))
    plt.imshow(y_fb); plt.title("Y (after feedback)")

    plt.subplot2grid((4, 2), (2, 1))
    plt.imshow(c2y_fb); plt.title("c_2(Y) (after feedback)")

    plt.subplot2grid((4, 2), (3, 0), colspan=1, rowspan=1)
    l1.showConvergenceImage(get=True); plt.title("Y Convergence")

    plt.subplot2grid((4, 2), (3, 1), colspan=1, rowspan=1)
    l1.showConvergenceMat(get=True); plt.title("Y Convergence")

    fig.tight_layout()


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
            visualize(L1, L2, X, Y, c_2Y, Y_fb, c_2Y_fb)
            plt.close('all')
            L1.unlock()
            L2.unlock()
            return
        L2.feed(L1.output_image, feedback=True); c_2Y_fb = L2.output_image
        visualize(L1, L2, X, Y, c_2Y, Y_fb, c_2Y_fb)
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



