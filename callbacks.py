from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context
import numpy as np
from sklearn.cluster import KMeans
import metrics

callbacks = keras.callbacks

class PrintACC(callbacks.Callback):
    def __init__(self, x, y, writer, freq=1):
        self.x = x
        self.y = y
        self.writer = writer
        self.freq = freq
        super(PrintACC, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) != 0:
            pass

        feature_model = Model(inputs=self.model.input, outputs=self.model.get_layer(name="embedding").output)
        features = feature_model.predict(self.x)
        km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
        y_pred = km.fit_predict(features)
        
        with context.eager_mode(), self.writer.as_default(), summary_ops_v2.always_record_summaries():
            for name, m in metrics.get_supervised_metric_handles():
                summary_ops_v2.scalar(
                    name,
                    m(self.y, y_pred),
                    step=epoch
                )
            for name, m in metrics.get_unsupervised_metric_handles():
                summary_ops_v2.scalar(
                    name,
                    m(features, y_pred),
                    step=epoch
                )


class ImageWriterCallback(callbacks.Callback):
    def __init__(self, ae, images, writer):
        self.writer = writer 
        self.ae = ae
        self.images = images
        self.scaled_images = np.array(self.images*255.0, dtype=np.uint8)
        self.n_channels = images.shape[-1] 
        self.make_summary(0, self.scaled_images, "original")

    def make_summary(self, step, tensor, type):
        with context.eager_mode(), self.writer.as_default(), summary_ops_v2.always_record_summaries():
            for i in range(self.n_channels):
                summary_ops_v2.image(
                    "%s image dim %d" % (type, i), 
                    tensor[:, :, :, i, tf.newaxis],
                    max_images=3,
                    step=step
                )

    def on_epoch_end(self, epoch, logs=None):
        with context.eager_mode(), self.writer.as_default(), summary_ops_v2.always_record_summaries():
            restored = self.ae.predict(self.images)
            self.make_summary(epoch, restored, "restored")