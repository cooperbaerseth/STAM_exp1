from STAM_classRepo import *

import matplotlib.gridspec as gridspec

def visualize(l1, l2, x, y, z, y_fb, z_prime):
    # Create subplot to visualize progression
    fig = plt.figure()
    gridspec.GridSpec(2, 4)

    plt.subplot2grid((2, 4), (0, 0))
    l1.show_STAMCentroids(2, 2, 5, 4); plt.axis('off')
    #plt.imshow(x); plt.title("X"); plt.axis('off')

    plt.subplot2grid((2, 4), (0, 1))
    rf_split(l1, x); plt.axis('off'); plt.title("X")

    plt.subplot2grid((2, 4), (0, 2))
    rf_split(l1, y); plt.axis('off'); plt.title("Y")

    plt.subplot2grid((2, 4), (0, 3))
    plt.imshow(z); plt.axis('off'); plt.title("Z")

    plt.subplot2grid((2, 4), (1, 0), colspan=1, rowspan=1)
    l1.showConvergenceImage(get=True); plt.axis('off'); plt.title("Y Convergence")

    plt.subplot2grid((2, 4), (1, 1), colspan=1, rowspan=1)
    rf_split(l1, l1.x_prime); plt.axis('off'); plt.title("X'")

    plt.subplot2grid((2, 4), (1, 2))
    rf_split(l1, y_fb); plt.axis('off'); plt.title("Y'")

    plt.subplot2grid((2, 4), (1, 3))
    plt.imshow(z_prime); plt.axis('off'); plt.title("Z'")

    fig.tight_layout()
    fig.suptitle('STAM Feedback', size='x-large', weight='bold')

    plt.pause(0.005)
    raw_input('Press Enter to exit')

def feedback(x, L1, L2):
    # x: original input image
    # n: number of feedback iterations

    L1.feed(x); X = L1.input_image
    L2.feed(L1.output_image); Y = L2.input_image; Z = L2.output_image
    L1.feed(L2.output_image, feedback=True); Y_prime = L1.output_image
    L2.adjustCent_feedback(L1.x_prime);Z_prime = L2.STAMs[0][0].centroids[L2.STAMs[0][0].prev_out]
    visualize(L1, L2, X, Y, Z, Y_prime, Z_prime)
    plt.close('all')
    L1.unlock()
    L2.unlock()

    return

'''
**********************
*********MAIN*********
**********************
'''

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize centroids
# centroids_init_c2avg = np.zeros((NUM_OF_CLUSTERS, x_train[0].shape[0], x_train[0].shape[0]))
# centroids_init_c2avg = initCents_close2avg(centroids_init_c2avg, x_train, y_train)

l1_perClass = 2
l2_perClass = 2
l1_centroids_init_rands, l2_centroids_init_rands = initCents_randomHierarchy(l1_perClass, l2_perClass, x_train, y_train)

# Initialize Layers
# l1 = Layer("L1", 7, 7, 0.005, centroids_init_c2avg)
# l2 = Layer("L2", 28, 28, 0.5, centroids_init_c2avg)

l1 = Layer("L1", 7, 7, 10*l1_perClass, 0.005, l1_centroids_init_rands)
l2 = Layer("L2", 28, 28, 10*l2_perClass, 0.005, l2_centroids_init_rands)

showInitCentroids(l1, 5, 4)
showInitCentroids(l2, 5, 4)

for i in range(x_train.shape[0]):
    feedback(x_train[i], l1, l2)



