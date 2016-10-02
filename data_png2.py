import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('lena_color.png')
print 'img shape', img.shape
plt.imshow(img)
plt.show()

img = img.transpose(2, 0, 1)
img = img.reshape(1, 3, 512, 512)
print 'type=',type(img), ', shape=', img.shape, ', ndim=', img.ndim
print 'data type=', img.dtype.name
print 'item size=', img.itemsize, ', size=', img.size

