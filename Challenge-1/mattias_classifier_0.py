# Modified https://elitedatascience.com/keras-tutorial-deep-learning-in-python
import tensorflow as tf
import matplotlib.pyplot as plt

# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibilityq

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from tensorflow.python.keras.datasets import mnist

# 4. Load the lego-jungle data
x = np.load('lego_class_data.npy')
y = np.load('lego_class_labels.npy')

# 5. Preprocess input data
X_test = np.expand_dims((x[:,:,:1000].astype('float32') / 255), axis=3).transpose((2,0,1,3))
X_train = np.expand_dims((x[:,:,1000:].astype('float32') / 255), axis=3).transpose((2,0,1,3))
Y_test = y[:1000,0]
Y_train = y[1000:,0]    # First column marks if data existseru

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(Y_train.astype('uint8'),2)
Y_test = np_utils.to_categorical(Y_test.astype('uint8'),2)

# 7. Define model architecture

model = Sequential()

model.add(Convolution2D(8, 3, 1, activation='relu', input_shape=(100,100,1),padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Convolution2D(16, 3, 1, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Convolution2D(32, 3, 1, activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# 8. Compile model
ada = optimizers.Adam(lr = 0.0003)
model.compile(loss='categorical_crossentropy',
              optimizer = ada,
              metrics=['accuracy'])

# 9. Fit model on training data
history = model.fit(X_train, Y_train,
          batch_size=32, epochs=100, verbose=1, validation_data = (X_test, Y_test))

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
plt.plot(history.history['val_acc'])
plt.show()

# 11. Our scoring
print('This network scored an average val_acc of ' + str(np.mean(history.history['val_acc'][-10:])) + ' in the last 10 epochs')
