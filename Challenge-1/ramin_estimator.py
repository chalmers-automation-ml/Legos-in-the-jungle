# Inspired by: https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

import tensorflow as tf
import matplotlib.pyplot as plt

# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
# from keras import optimizers
# from keras.optimizers import SGD
from tensorflow.python.keras.datasets import mnist

# 4. Load the lego-jungle data
x = np.load('./lego_est_data.npy')
y = np.load('./lego_est_labels.npy')

# 5. Preprocess input data
X_test = np.expand_dims((x[:,:,:1000].astype('float32') / 255), axis=3).transpose((2,0,1,3))
X_train = np.expand_dims((x[:,:,1000:].astype('float32') / 255), axis=3).transpose((2,0,1,3))
Y_test = y[:1000,1:]
Y_train = y[1000:,1:]    # Second-Fourth colums = rotation, x, y (or is it y,x?)
 
# 7. Define model architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(100,100,1)))
model.add(Convolution2D(64, 2, 2, activation='relu', padding = 'same'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 2, 2, activation='relu', padding = 'same'))
model.add(MaxPooling2D((2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 2, 2, activation='relu', padding = 'same'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 2, 2, activation='relu', padding = 'same'))
model.add(MaxPooling2D((2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 2, 2, activation='relu', padding = 'same'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 2, 2, activation='relu', padding = 'same'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 2, 2, activation='relu', padding = 'same'))
model.add(MaxPooling2D((2,2)))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# # model.add(MaxPooling2D((2,2)))

# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2,2)))

# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

# 8. Compile model
# model.compile(loss='mean_squared_error',
#               optimizer='Adam',
#               metrics=['accuracy'])
ada = optimizers.Adam(lr = 0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer = ada,
              metrics=['accuracy'])
 
# 9. Fit model on training data
history = model.fit(X_train, Y_train, 
          batch_size=32, epochs=100, verbose=1, validation_data = (X_test, Y_test))
 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print score

# 11. Our scoring
print('This network scored an average val_acc of ' + str(np.mean(history.history['val_acc'][-10:])) + ' in the last 10 epochs')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(history.history['val_acc'])
plt.title(str(np.mean(history.history['val_acc'][-10:])))
# ax.legend()
# plt.show()
 
fig.savefig('Ramin_Solution_Estimator.png')



