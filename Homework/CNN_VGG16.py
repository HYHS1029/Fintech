# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:01:14 2020

@author: Hsin-Yuan
"""
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
import cv2
#%% load data
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
#%% reshape the dataset (adding a color dimension) and convert from integers to floats
trainX=[cv2.cvtColor(cv2.resize(i,(48, 48)), cv2.COLOR_GRAY2BGR) for i in trainX]
testX=[cv2.cvtColor(cv2.resize(i,(48, 48)), cv2.COLOR_GRAY2BGR) for i in testX]
trainX=np.concatenate([arr[np.newaxis]for arr in trainX])
testX=np.concatenate([arr[np.newaxis]for arr in testX])

trainX = trainX.astype("float32")/255
trainX = trainX.reshape((60000, 48, 48, 3))
testX = testX.astype("float32")/255
testX = testX.reshape((10000, 48, 48, 3))
#%%
# do onehot encoding
trainY_onehot = to_categorical(trainY)
testY_onehot = to_categorical(testY)


#%% VGG-16 model
model = Sequential()
model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(48,48,3),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
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














