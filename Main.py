from Model import *
from TrainedModelTest import *


def main():
    # TODO: 8. Compile Model
    # model.compile(optimizer=SGD(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    #
    # TODO: 9. Fit Model
    # model.fit(x=x_train, y=y_train, epochs=1, batch_size=32)
    #
    # TODO: 10. Save Model to H5 File
    # model.save(filepath="ImageClassifierModel.h5")

    # TODO: 11. Evaluate Previously Trained Model
    # print(f"Test Loss: {results[0]}")
    # print(f"Test Accuracy: {results[1]}")

    # TODO: 12. Predict Previously Trained Model
    print(f"Prediction: {labels_array[max_prediction_index]}")


if __name__ == "__main__":
    main()

