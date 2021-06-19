from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

## Para normalizar el histograma -> dividir por la suma para tener valores entre 0 y 1
## np.histogram(A), np.histogram(B)

def loadImages():
    ct = Image.open('CT.pgm')
    mr = Image.open('MR.pgm')
    return np.array(ct), np.array(mr)


def showImages(a, b, title=None):
    plt.imshow(np.hstack([a, b]), cmap='gray')
    plt.title(title)
    plt.show()


def joint_hist(A, B, bDisplay=False):
    L = 255  # niveles de gris
    H, xedges, yedges = np.histogram2d(A.flatten(), B.flatten(), density=True, bins=L)

    sum_axis_0 = np.sum(H, axis=0)
    sum_axis_0 /= np.sum(sum_axis_0)

    sum_axis_1 = np.sum(H, axis=1)
    sum_axis_1 /= np.sum(sum_axis_1)

    if bDisplay:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(np.log(H+0.001), extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'))
        plt.colorbar()

        plt.xlabel('x')
        plt.ylabel('y')

        plt.plot(sum_axis_0)
        plt.show()

        plt.plot(sum_axis_1)
        plt.show()

    return sum_axis_0, sum_axis_1, H


def ssd(A, B):
    return np.linalg.norm((A-B).ravel()) / np.product(A.shape)


def mutual_information(A, B):
    hA, hB, hAB = joint_hist(A, B, bDisplay=False)
    suma = 0

    for i in range(0, hA.shape[0]):
        for j in range(0, hB.shape[0]):
            if hA[i] * hB[j] != 0:
                suma += (hAB[i][j] * np.log(hAB[i][j] / (hA[i] * hB[j]) + 0.00001))
    return suma # mutual information (un número real)


def evaluate_objective_function(A, B, objective_func):
    txMin, txMax, txInc = -10, 10, 1
    tyMin, tyMax, tyInc = txMin, txMax, txInc
    nx = int((txMax - txMin) / txInc + 1)
    ny = nx

    tx_candidates = np.linspace(txMin, txMax, num=nx, endpoint=True)
    ty_candidates = np.linspace(tyMin, tyMax, num=ny, endpoint=True)
    objective = np.zeros((tx_candidates.shape[0], ty_candidates.shape[0]))
    for i, tx in enumerate(tx_candidates):
        # print(i,tx)
        for j, ty in enumerate(ty_candidates):
            # print(j, ty)
            Bt = ndi.shift(B, (-tx, -ty))
            # showImages(B,Bt,title="tx="+str(tx)+", ty="+str(ty))
            objective[i, j] = objective_func(A, Bt)

    return objective, tx_candidates, ty_candidates


def plot_objective(objective, tx_values, ty_values):
    plt.imshow(np.log(objective), cmap='jet', interpolation='none')
    plt.xticks(range(len(tx_values)), tx_values, rotation='vertical')
    plt.yticks(range(len(ty_values)), ty_values)
    plt.title("objective function")
    plt.show()

    plt.figure()
    plt.contourf(np.log(objective), levels=30)  # ,cmap='RdGy')
    plt.xticks(range(len(tx_values)), tx_values, rotation='vertical')
    plt.yticks(range(len(ty_values)), ty_values)
    plt.title("iso-contour of objective function")
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    TX_values, TY_values = np.meshgrid(tx_values, ty_values)
    surf = ax.plot_surface(TX_values, TY_values, np.log(objective), cmap=cm.coolwarm)
    ax.contour(TX_values, TY_values, np.log(objective), levels=50)

    # Customize the z axis.
    # ax.set_zlim(0,np.max(mi.ravel()))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.title("objective as surface")
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def quadratic_subpixel(values):
    # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    '''
    tu código
    '''
    return # un número real


def refine_subpixel_precision(objective, peak_x, peak_y, tx_values, ty_values):
    sx = quadratic_subpixel(...) # 1D para x
    sy = quadratic_subpixel(...) # 1D para y
    return # dos números real

def test_histogramas_conjuntos(a, b):
    joint_hist(a, a, bDisplay=True)
    joint_hist(a, ndi.shift(a,(1,1)), bDisplay=True)
    joint_hist(a, ndi.shift(a,(5,5)), bDisplay=True)
    joint_hist(a, ndi.shift(a,(10,10)), bDisplay=True)
    joint_hist(a, ndi.shift(a,(30,30)), bDisplay=True)
    joint_hist(a, b, bDisplay=True)

if __name__ == "__main__":
    ct, mr = loadImages()
    showImages(ct, mr, "Original images")

    a = ct
    b = mr

    # para el caso sintético

    inicio = time()
    bTransfSintetica=True
    if bTransfSintetica:
        true_tx, true_ty = -6.45, 3.75 # los valores sintéticos deseados; probar varios
        b = ndi.shift(a, (true_tx, true_ty))

    minimization_funcs = [ssd]  # since we will maximize, we need to know which functions are "cost" functions
    # para usar SSD o MI
    ofunc = ssd  # objective function to use
    #ofunc = mutual_information  # objective function to use

    # evaluate the objective function at candidate translations
    objective, tx_values, ty_values = evaluate_objective_function(a, b, objective_func=ofunc)

    # display the objective function (we can do that because we have only 2 motion parameters)
    #plot_objective(objective, tx_values, ty_values)

    if ofunc in minimization_funcs:
        objective *= -1  # for 'cost' functions (i.e. the lower the better), we invert the values for maximization

    print(peak_local_max(objective))
    # finding maximum with pixel accuracy
    peak_x, peak_y = peak_local_max(objective)[0]
    tx, ty = tx_values[peak_x], ty_values[peak_y]

    if bTransfSintetica:
        print("true translation (tx,ty):", true_tx, true_ty)

    print("estimated translation (tx,ty):", tx, ty)

    # get subpixel accuracy
    #tx, ty = refine_subpixel_precision(objective, peak_x, peak_y, tx_values, ty_values)
    #print("estimated subpixel precision translation (tx,ty):", round(tx, 2), round(ty, 2))

    # display images before and after registration
    alpha = 0.5
    imgs_before_registering = alpha * a + (1 - alpha) * b
    b_shifted = ndi.shift(b, (-tx, -ty))
    imgs_after_registering = alpha * a + (1 - alpha) * b_shifted

    final = time()
    print("Tiempo empleado: {0:.2f} ".format(final - inicio))
    showImages(imgs_before_registering, imgs_after_registering, "Images overlapped before (left) and after (right) registration")