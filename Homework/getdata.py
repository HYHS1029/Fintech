# import the necessary packages
from keras.datasets import fashion_mnist



print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()