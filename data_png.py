import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('lena.png')
print('img shape', img.shape)
plt.imshow(img, cmap='bone')
plt.show()

img = img.reshape(1,1,512,512)
print('type=',type(img), ', shape=', img.shape, ', ndim=', img.ndim)
print('data type=', img.dtype.name)
print('item size=', img.itemsize, ', size=', img.size)
