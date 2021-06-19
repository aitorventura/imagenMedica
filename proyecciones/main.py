import numpy as np
import matplotlib.pyplot as plt

from skimage.data import immunohistochemistry, retina, microaneurysms
from skimage.transform import radon, rescale, iradon
# from scipy.fftpack import fft, fftshift
# from skimage.measure import profile_line
from skimage.color import rgb2gray

def circle(m, n, r):
    im = np.zeros((m, n))
    y, x = np.meshgrid(np.linspace(0, m, m), np.linspace(0, n, n))
    y0, x0 = m / 2, n / 2
    dY, dX = y - y0, x - x0
    return np.sqrt(dY * dY + dX * dX) < r

def twocircles(m, n, r1, r2):
    circ1 = circle(int(0.6 * m), n, r1)
    circ2 = circle(int(0.4 * m), n, r2) * 0.5
    return np.hstack((circ1, circ2))

def get_medical_image(img_name):
    return rgb2gray(eval(img_name)())

def projection(im, angle):
    return radon(im, [angle]).sum(axis=1)

def sinogram(im, angles=None):
    if angles is None:
        angles = np.linspace(0, 180, 180)
    return (radon(im, angles), angles)

def plot_sinogram(sinogram_im, angles):
    print(sinogram_im.shape)
    # print(angles)
    num_rhos, num_angles = sinogram_im.shape
    plt.imshow(sinogram_im)

    pos_angles = np.linspace(0, num_angles, 8, endpoint=False).astype(int)
    plt.xticks(pos_angles, angles[pos_angles].astype(int))

    pos_rhos = np.linspace(0, num_rhos, 8, endpoint=False).astype(int)
    plt.yticks(pos_rhos, (pos_rhos - num_rhos / 2).astype(int))
    plt.xlabel('theta [degrees]')
    plt.ylabel('rho')
    plt.show(block=True)

medical_imgs_names = ['retina', 'immunohistochemistry', 'microaneurysms']
medical_images = {img_name: get_medical_image(img_name) for img_name in medical_imgs_names}
w = h = 200
R1,R2 = 40,20
im = circle(w, h, R1)
im2 = twocircles(w, h, R1, R2)
images = {'im': im, 'im2': im2}
images.update(medical_images)
plt.set_cmap('gray')

# Calcular proyecciones individuales
def proyecciones():
    for im_name in ['im', 'im2']:
        im = eval(im_name)
        plt.imshow(im)
        plt.show(block=True)
        plt.title('image: ' + im_name)
        for angle in [0, 90, 45, 10]:
            radon_proj = radon(im, [angle])
            plt.plot(radon_proj)
            plt.title('image: ' + im_name + ", angle (degrees): " + str(angle))
            plt.show(block=True)

# Obtener reconstrucción
def reconstruccion(n):
    for im_name in ['im', 'im2']:
        im = eval(im_name)
        thetas = np.linspace(0, 180, n, endpoint=False)
        radon_proj = radon(im, thetas)
        plt.imshow(radon_proj)
        plt.show(block=True)
        plt.title("image :" + im_name + "angles: " + str(thetas))
        plt.imshow(iradon(radon_proj, thetas, filter_name='ramp', circle=False))
        plt.show(block=True)


# Visualizar sinograms
def sinogramas():
    plot_sinogram(*sinogram(im))
    plot_sinogram(*sinogram(im2))

# Filtered backprojection
def retroproyecciones(n):
    thetas = np.linspace(0, 180, n, endpoint=False)
    for image_name, im in images.items():
        plt.imshow(im)
        plt.show(block=True)
        radon_proj = radon(im, thetas)
        for filter in ['ramp']:
            plt.imshow(radon_proj)
            plt.title("sinogram of image " + image_name)
            plt.pause(3)
            plt.show(block=True)
            plt.imshow(iradon(radon_proj, thetas, circle=False))
            plt.title("Reconstruction of image: " + image_name + " with iradon with filter: " + str(filter))
            plt.show(block=True)

if __name__ == "__main__":
    #proyecciones()
    #sinogramas()
    #reconstruccion(180)
    retroproyecciones(90)



