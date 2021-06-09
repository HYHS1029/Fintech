# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 01:24:19 2020

@author: Hsin-Yuan
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras import models 
#%% load data
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
#%% reshape the dataset (adding a color dimension) and convert from integers to floats
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype('float32')
testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype('float32')
# normalize the image to range 0-1
trainX /= 255.0
testX /= 255.0
#%%
# do onehot encoding
trainY_onehot = to_categorical(trainY)
testY_onehot = to_categorical(testY)

#%% Define the CNN　Model
model = Sequential()  
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (28, 28, 1), strides=(1, 1), kernel_initializer='he_normal' ,activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
# Second convolutional layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))
# Third convoluyional layer
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim = 10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#%%　fitting the model
train_history = model.fit(trainX, trainY_onehot, validation_split = 0.2, batch_size= 128, epochs = 25)

#%% 2. plot
plt.figure()
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.figure()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('learning curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
#%% evaluate the test prediction accuracy
score = model.evaluate(testX, testY_onehot)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%% 3. plot activations of the ﬁrst layer
img = testX[:1]
plt.figure()
plt.imshow(img[0], cmap = 'gray_r')
# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in model.layers[:12]] 
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs = model.input, outputs = layer_outputs) 
# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img)
first_layer_activation = activations[0]
plt.figure(figsize=(12, 12))
for i in range(1, 33):
    plt.subplot(4, 8, i)
    plt.imshow(first_layer_activation[0, :, :, i-1], cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
plt.show()

#%% 4. Visualize prediction
# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

y_hat = model.predict(testX)
# Plot a random sample of 10 test images, their predicted labels and ground truth
fig = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(testX.shape[0], size=15, replace=False)):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(testX[index]), cmap='gray_r')
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(testY_onehot[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))  
#%%








