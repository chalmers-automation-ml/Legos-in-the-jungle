# Modified https://elitedatascience.com/keras-tutorial-deep-learning-in-python
import tensorflow as tf

# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.python.keras.datasets import mnist
 
# 4. Load the lego-jungle data
x = np.load('F:\Dropbox\ml_course\legos\legos\lego_est_data.npy')
y = np.load('F:\Dropbox\ml_course\legos\legos\lego_est_labels.npy')

# 5. Preprocess input data
X_test = np.expand_dims((x[:,:,:1000].astype('float32') / 255), axis=3).transpose((2,0,1,3))
X_train = np.expand_dims((x[:,:,1000:].astype('float32') / 255), axis=3).transpose((2,0,1,3))
Y_test = y[:1000,1:]
Y_train = y[1000:,1:]    # Second-Fourth colums = rotation, x, y (or is it y,x?)
 
 
# 7. Define model architecture

model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(100,100,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='relu'))
 
# 8. Compile model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, epochs=50, verbose=1, validation_data = (X_test, Y_test))
 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

# 11. Our scoring
print('This network scored an average val_acc of ' + str(np.mean(history.history['val_acc'][-10:])) + ' in the last 10 epochs')
