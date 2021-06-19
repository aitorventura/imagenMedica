import matplotlib.pylab as plt
import numpy as np

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

fig, ax = plt.subplots()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
img = np.random.randint(0,3,(100,100))
ax.imshow(img,cmap='gray')

plt.show()


