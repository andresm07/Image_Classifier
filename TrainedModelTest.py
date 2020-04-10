from ImagesDataset import *
from keras.models import load_model
import numpy as np

model = load_model('ImageClassifierModel.h5')
results = model.evaluate(x=x_test, y=y_test)

test_image_data = np.asarray([x_test[0]])

prediction = model.predict(x=test_image_data)
max_prediction_index = np.argmax(prediction[0])

