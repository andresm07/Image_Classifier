from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm

# TODO: 1. Initialize model
model = Sequential()

# TODO: 2. Create Convulutional Layer
# filters: amount of outputs
# kernel_size: size of matrix filter
# padding: shape of output
# kernel_constraint: max value of kernel operation
convolutional_layer = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu", padding="same",
                             kernel_constraint=maxnorm(3))
model.add(convolutional_layer)

# TODO: 3. Create Max Pool Layer
max_pooling_layer = MaxPooling2D(pool_size=(2, 2))
model.add(max_pooling_layer)

# TODO: 4. Create Flatten Layer
flatten_layer = Flatten()
model.add(flatten_layer)

# TODO: 5. Create Internal Dense Layer
# units: the higher amount of units, the more accurate the model, but the more training time it needs
internal_dense_layer = Dense(units=512, activation="relu", kernel_constraint=maxnorm(3))
model.add(internal_dense_layer)

# TODO: 6. Create Dropout Layer
# rate: rate at which model drops neurons
dropout_layer = Dropout(rate=0.5)
model.add(dropout_layer)

# TODO: 7. Create External Dense Layer
external_dense_layer = Dense(units=10, activation="softmax")
model.add(external_dense_layer)