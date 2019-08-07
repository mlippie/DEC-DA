from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context

class SobelMSELossCallback(keras.callbacks.Callback):

    def __init__(self, loss, writer, steps_per_epoch, freq="epoch"):
        self.loss = loss
        self.writer = writer
        self.steps_per_epoch = steps_per_epoch

        if freq == "epoch":
            self.freq = steps_per_epoch
        else:
            self.freq = freq
        self.reset()
    
    def summarize(self):
        with context.eager_mode(), self.writer.as_default(), summary_ops_v2.always_record_summaries():
            for i in range(sobel_true.get_shape()[-1]):
                summary_ops_v2.image(
                    "%s image dim %d" % (type, i), 
                    sobel_true[:, :, :, i, 0],
                    max_images=3
                )

            summary_ops_v2.scalar(
                "Mean squared reconstruction error",
                self.epoch_total_mse / self.step
            ) 
            summary_ops_v2.scalar(
                "Mean squared sobel error",
                self.epoch_total_msse / self.step
            )

    def reset(self):
        self.step = 0
        self.epoch_total_mse = 0
        self.epoch_total_msse = 0

    def on_batch_end(self, step, logs=None):
        print(self.loss.mse.numpy())
