# import tensorflow
# import keras
#
# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Dense, Dropout
# from keras.optimizers import SGD
#
# model = Sequential()
#
# conv_layer = Conv2D(filter=32, kernel_size=(3, 3), activation='relu')
# max_pool_layer = MaxPooling2D(pool_size=(2, 2))
# dense_layer = Dense(1024, activation='softmax')
# dropout_layer = Dropout(rate=0.5)
#
# model.add(conv_layer)
# model.add(max_pool_layer)
# model.add(dense_layer)
# model.add(dropout_layer)
# model.add(dense_layer)
#
# model.compile(optimizer=SGD, loss=0.01, metrics=['accuracy'])

from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras.utils as utils
import numpy as np


def reshape_image(input_image_arrays):
    output_array = []
    for image_array in input_image_arrays:
        output_array.append(image_array.reshape(-1))
    return np.asarray(output_array)


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# print(train_images[0])
# plt.imshow(train_images[0])
# plt.show()
# max_index = np.argmax(train_labels[0])
# print(labels_array[max_index])


labels_array = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

train_images = train_images.astype("float32")
train_images = train_images / 255.0

test_images = test_images.astype("float32")
test_images = test_images / 255.0
