from STAM_classRepo import *



def feedback(x, n, L1, L2):
    # x: original input image
    # n: number of feedback iterations

    L1.feed(x)
    L2.feed(L1.output_image)
    for i in range(n):
        L1.feed(L2.output_image)
        if L1.converged():
            L1.unlock()
            return
        L2.feed(L1.output_image)
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


