# https://wiseodd.github.io/techblog/2016/11/05/levelset-method/
# https://wiseodd.github.io/techblog/2016/11/20/levelset-segmentation/

from skimage import measure
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io
from mpl_toolkits.mplot3d import Axes3D


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x)) ** 2)


def default_phi(x):
    # Initialize surface phi at the border (margin px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    margin = 15
    phi[margin:-margin, margin:-margin] = -1
    return phi


def show_fig1(fig1, phi):
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)


def show_fig2(fig2, phi, img):
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)


def naive_levelSets(img, img_smooth):
    # naive level sets

    F = stopping_fun(img_smooth)
    plt.imshow(F, cmap='gray')
    plt.show()

    phi = default_phi(img)
    dt = 1.

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    show_fig1(fig1, phi)
    show_fig2(fig2, phi, img)

    n_iter = 250
    for i in range(n_iter):
        dphi = grad(phi)
        dphi_norm = norm(dphi)

        dphi_t = F * dphi_norm

        phi = phi + dt * dphi_t

        # plt.imshow(img)
        # print("zeros:",np.sum(phi>0))
        # plt.imshow(np.linalg.norm(dphi),cmap='gray')
        show_fig1(fig1, phi)
        show_fig2(fig2, phi, img)
        plt.pause(0.5)

    print("iters:", n_iter)


# geodesic level sets


def curvature(f):
    fy, fx = grad(f)
    norm = np.sqrt(fx ** 2 + fy ** 2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)


def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)


def geodesic_levelSets(img, img_smooth):
    print("Now the geodesic level sets")

    phi = default_phi(img)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    show_fig1(fig1, phi)
    show_fig2(fig2, phi, img)

    v = 1.
    dt = 1.
    # alpha=0.3
    g = stopping_fun(img_smooth)  # , alpha)
    dg = grad(g)
    n_iter = 130
    for i in range(n_iter):
        print(i)
        dphi = grad(phi)
        dphi_norm = norm(dphi)
        kappa = curvature(phi)

        smoothing = g * kappa * dphi_norm
        balloon = g * dphi_norm * v
        attachment = dot(dphi, dg)

        dphi_t =  balloon + attachment

        phi = phi + dt * dphi_t

        show_fig2(fig2, phi, img)
        plt.pause(0.5)
        '''
        plt.imshow(phi,cmap='gray')
        plt.show()
        plt.pause(0.5)
        '''


if __name__ == "__main__":
    # read image, convert to gray levels and normalize
    img = io.imread('./imgs/twoObj.bmp')
    img = color.rgb2gray(img)
    img = img - np.mean(img)

    print(img)
    gx, gy = np.gradient(img)

    print(gx)
    plt.imshow(gx, cmap='gray', interpolation='none')
    plt.show()

    plt.imshow(gy, cmap='gray', interpolation='none')
    plt.show()

    # Smooth the image to reduce noise and separation between noise and edge becomes clear
    sigma = 0
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

    # evolve the surface with the naive approach and/or the GAC formulationi

    bNaive = False
    if bNaive:
        naive_levelSets(img, img_smooth)

    bGAC = True
    if bGAC:
        geodesic_levelSets(img, img_smooth)
