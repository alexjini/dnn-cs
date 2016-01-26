import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

img = mpimg.imread('../prep-dnn-lecture/figure/lena_color.png')
print 'img shape', img.shape
plt.imshow(img)
plt.show()

img = img.transpose(2, 0, 1)
img = img.reshape(1, 3, 512, 512)
print 'type=',type(img), ', shape=', img.shape, ', ndim=', img.ndim
print 'data type=', img.dtype.name
print 'item size=', img.itemsize, ', size=', img.size

b = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

model = Sequential()
model.add(Convolution2D(2, 3, 3, border_mode='valid',
                        input_shape=(3, 512, 512)))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='mse', optimizer='sgd')
hist = model.fit(img, b, batch_size=1, nb_epoch=2, show_accuracy=True, verbose=1)
print hist.history['acc']
