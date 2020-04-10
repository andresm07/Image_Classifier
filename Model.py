from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm

# TODO: 1. Initialize model as a sequential type to add layers in order
model = Sequential()

# TODO: 2. Create Convulutional Layer
# Add the first convolution to output a feature map
# filters: amount of outputs
# kernel_size: size of matrix filter used to calculate output features
# input_shape: each image is 32x32x3
# activation: relu activation for each of the operations as it produces the best outputs
# padding: shape of output -> "same" adds padding to the input image to make sure that the output feature map
#          is the same size as the input
# kernel_constraint: maxnorm normalizes the values in the kernel to make sure that the max value of kernel operation
#                    is 3
convolutional_layer = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu", padding="same",
                             kernel_constraint=maxnorm(3))
model.add(convolutional_layer)

# TODO: 3. Create Max Pool Layer
# Add a max pool layer to decrease the image size from 32x32 to 16x16
# pool_size: finds the max value in each 2x2 section of the input
max_pooling_layer = MaxPooling2D(pool_size=(2, 2))
model.add(max_pooling_layer)

# TODO: 4. Create Flatten Layer
# Flatten layer converts the matrix into a one dimensional array
flatten_layer = Flatten()
model.add(flatten_layer)

# TODO: 5. Create Internal Dense Layer
# First dense layer to create the actual prediction network
# units: the higher amount of units, the more accurate the model, but the more training time it needs
#        512 neurons at this layer shows good results
internal_dense_layer = Dense(units=512, activation="relu", kernel_constraint=maxnorm(3))
model.add(internal_dense_layer)

# TODO: 6. Create Dropout Layer
# Dropout layer to ignore some neurons during the training which improves model reliability
# rate: rate at which model drops neurons
dropout_layer = Dropout(rate=0.5)
model.add(dropout_layer)

# TODO: 7. Create External Dense Layer
# Final dense layer used to produce output for each of the 10 categories 
# units: 10 categories in CIFAR-10, so 10 outputs
# activation: "softmax" because we are calculating probabilities for each of the 10 categories
external_dense_layer = Dense(units=10, activation="softmax")
model.add(external_dense_layer)