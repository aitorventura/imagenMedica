import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters


def onclick(event):
    contour.append([event.xdata, event.ydata])

    if event.dblclick:
        plt.close()


initial_contours = {'two': [56, 30, 15],
                    'eco': [345, 255, 50]}  # [x,y,r] where (x,y) is the center and r is the radius of a circumference
image_name = {'two': 'twoObj.bmp', 'eco': 'ECO.pgm'}
config_data = {'two': [2, 5, 15], 'eco': [3, 3, 3]}  # [num_neighbours, alpha, beta]
x_y_lims = {'two': ([20, 90], [10, 50]), 'eco': ([100, 500], [100, 400])}

# select image to use
img_key = 'eco'
img_key = 'two'

# open the selected image
path = 'imgs/'
I = Image.open(path + image_name[img_key])

# convert image object to Numpy array
I = np.array(I)

# convert to float to prevent numerical issues later on
I = I.astype(float)

apply_filter = False
if apply_filter:
    copy = np.copy(I)
    sigma = 0
    I = filters.gaussian(I, sigma=sigma)
    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(copy)
    plt.title("Original")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(I)
    plt.title("With gaussian {}".format(sigma))

    plt.show()
gradiente = filters.sobel(I)

# default image limits
xmin, ymin = 0, 0
ymax, xmax = I.shape[0] - 1, I.shape[1] - 1

# set image limits to crop image at visualisation stage
bLimits = True  #
if bLimits:
    xmin, xmax = x_y_lims[img_key][0]
    ymin, ymax = x_y_lims[img_key][1]

# data for initial contour
x0, y0, R = initial_contours[img_key]

# data for energy minimisation
neig, alpha, beta = config_data[img_key]

# Contour (curve) initialisation
Npoints = 21
N = Npoints + 1
Qx = np.zeros(N)
Qy = np.zeros(N)

num_points = Npoints

contour_circe = True
contour = []
manual_contour = False

if manual_contour:
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.imshow(I, interpolation='none', cmap='gray')
    plt.show()

    Npoints = len(contour)
    num_points = len(contour)
    N = Npoints
else:
    if contour_circe:
        for theta in np.linspace(0, 2 * np.pi, num_points, endpoint=False):
            x, y = x0 + R * np.cos(theta), y0 + R * np.sin(theta)
            contour.append([x, y])
        contour.append([0, 0])
    else:
        contour.append([40, 40])
        contour.append([45, 40])
        contour.append([50, 40])
        contour.append([55, 40])
        contour.append([60, 40])
        contour.append([65, 40])
        contour.append([70, 40])
        contour.append([70, 35])
        contour.append([70, 30])
        contour.append([70, 25])
        contour.append([70, 20])
        contour.append([70, 15])
        contour.append([65, 15])
        contour.append([60, 15])
        contour.append([55, 15])
        contour.append([50, 15])
        contour.append([45, 15])
        contour.append([40, 15])
        contour.append([40, 20])
        contour.append([40, 25])
        contour.append([40, 30])
        contour.append([40, 35])

print("contour has ", len(contour), "points")

# from list to arrays
Qx = np.array([e[0] for e in contour])
Qy = np.array([e[1] for e in contour])

# close curve
Qx[N - 1] = Qx[0]
Qy[N - 1] = Qy[0]

# print contour points
print("Countour points:")
for k in range(Qx.shape[0]):
    print(Qx[k], Qy[k])

energias = []
# evolve the curve
num_iters = 40
for it in range(num_iters):
    # display the image and the countour
    if not manual_contour:
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    plt.imshow(I, interpolation='none', cmap='gray')
    plt.plot(Qx, Qy, marker='o', markerfacecolor='orange', linestyle='dashed')
    plt.pause(0.5)

    Rx = np.copy(Qx)
    Ry = np.copy(Qy)

    # iterate over the contour points
    for k in range(N):

        x = Qx[k]
        y = Qy[k]

        min = np.inf

        x_chosen = x
        y_chosen = y

        prev = k - 1
        next = k + 1

        if k == N - 1:
            next = 0
        if k == 0:
            prev = N - 1

        # look for local candidate points in a neighbourhood
        for i in range(-neig, neig + 1):
            for j in range(-neig, neig + 1):

                x_neig = int(x + i)
                y_neig = int(y + j)

                alphaEnergy = alpha * ((Qx[next] - x_neig) ** 2 + (Qy[next] - y_neig) ** 2)
                betaEnergy = beta * ((Qx[next] - 2 * x_neig + Qx[prev]) ** 2 + (Qy[next] - 2 * y_neig + Qy[prev]) ** 2)
                energyInt = alphaEnergy + betaEnergy

                energyExt = - (gradiente[y_neig][x_neig] ** 2)

                energy = energyInt + energyExt

                if energy < min:
                    min = energy
                    x_chosen = x_neig
                    y_chosen = y_neig

        Rx[k] = x_chosen
        Ry[k] = y_chosen

    #  Close curve
    Rx[N - 1] = Rx[0]
    Ry[N - 1] = Ry[0]

    # Update Qx y Qy
    Qx = np.copy(Rx)
    Qy = np.copy(Ry)

    energias.append(min)

    print('Iteration ', it)

ks = np.arange(1, num_iters + 1, 1)

plt.close()
plt.plot(ks, energias)
plt.show()
