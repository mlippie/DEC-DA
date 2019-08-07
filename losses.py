from tensorflow import keras, image, transpose, losses

class SobelMSELoss(losses.Loss):

    def __init__(self):
        super().__init__(name="SobelMSE")

    def call(self, y_true, y_pred, sample_weight=None):
        sobel_true = image.sobel_edges(y_true)
        sobel_pred = image.sobel_edges(y_pred)

        mse = keras.backend.mean(keras.backend.square(y_true - y_pred))
        msse = keras.backend.mean(keras.backend.square(sobel_true - sobel_pred))

        return msse + mse
