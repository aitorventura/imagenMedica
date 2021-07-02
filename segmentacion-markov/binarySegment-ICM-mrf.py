from scipy import misc
import numpy as np
import random
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io

# Source: https://gist.github.com/yadrimz/fb959cc45a69a3cc2d521ebd01215d5chttps://gist.github.com/yadrimz/fb959cc45a69a3cc2d521ebd01215d5c
# V. Javier Traver: Adapted from binary denoising to binary segmentation

xrange = range

# Given parameters of gray-level distributions for each region class
mean_gray_class = [75, 195]
stdev_gray_class = [5, 5]


# Conditional Gaussian distribution of gray level values for each class
def prob_gray_class(im, classLabel):
    # classLabel is 0 or 1
    return np.exp(-((im - mean_gray_class[classLabel]) ** 2) / (2 * stdev_gray_class[classLabel] ** 2))


# load input image to segment
img_file = 'monedas.png'
img_file = 'ECO.pgm'
img_file = 'twoObj.bmp'
path = 'imgs/'
im = io.imread(path + img_file)
im = np.asarray(im)

# image dimensions
H, W = im.shape
print(H, W)

# want to work with square image?
bSquare = False
if bSquare:
    side = min(H, W)
    im = np.asarray(im)[:side, :side]
    H, W = side, side

# convert to float numbers
im = im.astype(float)

# add noise to image for more challenging segmentations
scale_noise = 20  # amount of noise level (0 for no noise)
# choose between uniform or gaussian noise
noise_fn = np.random.randn  # gaussian
noise_fn = np.random.rand  # uniform
im = im + scale_noise * noise_fn(H, W)

# display original image with noise added
imgplot = plt.imshow(im, cmap=cm.Greys_r)
plt.title("original image")
plt.pause(1)

# initial segmentation (to initialize the ICM algorithm)
bProbInit = True
if bProbInit:  # initialize based on per-pixel probability
    probs = np.array([prob_gray_class(im, 0), prob_gray_class(im, 1)])
    segm_prev = 2 * np.argmax(probs, axis=0) - 1  # mapping class labels from {0,1} to {-1,1}
else:
    segm_prev = np.random.randint(-1, 1, (H, W))  # per-pixel random initialization

# to swap with segm_prev after each ICM iteration
segm = np.zeros_like(segm_prev)

# display the initial segmentation
imgplot = plt.imshow(segm_prev)
plt.title("Initial segmentation")
plt.pause(1)

# El término del nivel de gris para la función criterio
# Es un array de 3 dimensiones: 2 x m x n, y el primer índice (valores 0,1) representa la clase de la region (valores -1,1)
pixel_term = np.array(
    [((im - mean_gray_class[classLabel]) ** 2.0) / (2 * stdev_gray_class[classLabel] ** 2) for classLabel in [0, 1]])
alpha = 100  # peso del potencial unario
beta = 50  # peso del potencial binario

# mapping region label {-1,1} to index {0,1} the pixel-term array
pos_pixel_term = {-1: 0, 1: 1}

# Iterated Conditianl Modes (ICM) algorithm

next_iteration = True
iteration = 1
last_segmentation = segm_prev.copy()


def energy_one_function(segm_prev, alpha, beta, i, j):
    x_i = 1
    pairwise_factor = 0  # the cliques that represent pairs between latent variables (neighbours)
    if i - 1 >= 0:
        pairwise_factor += segm_prev[i - 1, j] * x_i
    if i + 1 < H:
        pairwise_factor += segm_prev[i + 1, j] * x_i
    if j - 1 >= 0:
        pairwise_factor += segm_prev[i, j - 1] * x_i
    if j + 1 < W:
        pairwise_factor += segm_prev[i, j + 1] * x_i
    energy_one = pixel_term[pos_pixel_term[x_i]][i, j] - alpha * x_i - beta * pairwise_factor
    return energy_one


def energy_minus_one_function(segm_prev, alpha, beta, i, j):
    x_i = -1
    pairwise_factor = 0
    if i - 1 >= 0:
        pairwise_factor += segm_prev[i - 1, j] * x_i
    if i + 1 < H:
        pairwise_factor += segm_prev[i + 1, j] * x_i
    if j - 1 >= 0:
        pairwise_factor += segm_prev[i, j - 1] * x_i
    if j + 1 < W:
        pairwise_factor += segm_prev[i, j + 1] * x_i
    energy_minus_one = pixel_term[pos_pixel_term[x_i]][i, j] - alpha * x_i - beta * pairwise_factor
    return energy_minus_one


while next_iteration:  # arbitrary stopping criterion; convergence-based criteria would be possibl

    # display segmentation at each iteration
    plt.imshow(segm_prev)
    plt.title("ICM, iteration: " + str(iteration))
    plt.pause(1)
    last_segmentation = segm_prev.copy()
    # iterate over the pixels

    for i in xrange(0, H):
        for j in xrange(0, W):

            # Try one choice

            energy_one = energy_one_function(segm_prev, alpha, beta, i, j)

            # Try the other choice

            energy_minus_one = energy_minus_one_function(segm_prev, alpha, beta, i, j)

            # Choose the choice of minimum energy
            if energy_one < energy_minus_one:
                segm[i, j] = 1
            else:
                segm[i, j] = -1

    error = np.mean(segm.copy() != last_segmentation)
    if error == 0:
        next_iteration = False

    iteration += 1

    segm_prev = segm  # update the segmentation only after the labels of *all* pixels have been updated

# display the final segmentation
imgplot = plt.imshow(segm_prev, cmap=cm.Greys_r)
plt.show(block=True)
