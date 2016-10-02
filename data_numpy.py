from keras.utils import np_utils
import numpy as np

a = np.array([0,0,1,1,2,2,1,1,3,3,0,0])
b = np.array([(0,1,2),(1,3,0)])

print 'a'
print 'type=',type(a), ', shape=', a.shape, ', ndim=', a.ndim
print 'data type=', a.dtype.name
print 'item size=', a.itemsize, ', size=', a.size

print 'b'
print 'type=',type(b), ', shape=', b.shape, ', ndim=', b.ndim
print 'data type=', b.dtype.name
print 'item size=', b.itemsize, ', size=', b.size

a = a.reshape(6,2)
b = b.reshape(6)

a = a.astype('float32')
c = np_utils.to_categorical(b, 4)

print 'a'
print 'type=',type(a), ', shape=', a.shape, ', ndim=', a.ndim
print 'data type=', a.dtype.name
print 'item size=', a.itemsize, ', size=', a.size

print 'c'
print 'type=',type(c), ', shape=', c.shape, ', ndim=', c.ndim
print 'data type=', c.dtype.name
print 'item size=', c.itemsize, ', size=', c.size

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

model = Sequential()
model.add(Dense(4, init='uniform', input_shape=(2,), activation='softmax'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
plot(model, to_file='model.png')

model.fit(a, c, batch_size=1, nb_epoch=3, verbose=1)
