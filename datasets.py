import numpy as np
import time
import h5py

try:
    import tensorflow as tf
except ImportError:
    pass


def logtrans_ssc(inp, chan):
    if chan == 'channel_6':
        return np.log(inp)


def minmaxnorm(inp):

    # per image normalization
    min_ = inp.reshape(inp.shape[0], -1).min(axis=1)
    max_ = inp.reshape(inp.shape[0], -1).max(axis=1)

    max_ = np.where(max_== 0.0, np.ones_like(max_), max_)

    return ((inp.T-min_)/(min_+max_)).T


def quantminmaxnorm(inp):

    # per image normalization
    quants = np.quantile(inp.reshape(inp.shape[0], -1), q=[0.05, 0.95], axis=1)
    quants = np.where(quants == 0.0, np.ones_like(quants), quants)

    return ((inp.T-quants[0])/(quants[0]+quants[1])).T


class DatasetWrapper:
    def __init__(self, h5fp, subset_key=None, channels=None, norm_func=minmaxnorm, trans_func=logtrans_ssc):
        if channels is None:
            channels = [k for k in list(h5fp.keys()) if "channel" in k]

        images_shape = list(h5fp[channels[0]]["images"].shape)
        ims = np.empty(shape=images_shape, dtype=np.float32)
        masks = np.empty(shape=images_shape, dtype=np.int8)

        if subset_key is not None:
            indices = h5fp["subsets"][subset_key][:]
            images_shape[0] = len(indices)
        else:
            indices = np.s_[:]

        self.labels = None
        if "labels" in h5fp:
            self.labels = h5fp["labels"][:][indices]

        shape = tuple([len(channels)] + images_shape)
        self.images = np.empty(shape=shape, dtype=np.float32)
        
        for i, chan in enumerate(channels):
            print("Loading %s" % chan)
            h5fp[chan]["images"].read_direct(ims)
            h5fp[chan]["masks"].read_direct(masks)

            self.images[i] = np.multiply(ims[indices], masks[indices], dtype=np.float32)

            if trans_func is not None:
                self.images[i] = trans_func(self.images[i], chan)
            
            if norm_func is not None:
                self.images[i] = norm_func(self.images[i])

        self.images = np.moveaxis(self.images, 0, -1)


def load_hdf5(dataset, subset_key=None):
    with h5py.File(dataset) as h5fp:  
        ds = DatasetWrapper(
            h5fp,
            subset_key
        )
    
    print("HDF5 samples", ds.images.shape, ds.labels.shape)
    return ds.images, ds.labels


def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_mnist_test():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    _, (x, y) = mnist.load_data()
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        raise ValueError("No data for usps found, please download the data from links in \"./data/usps/download_usps.txt\".")

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y


def load_data_conv(dataset, subset_key=None):
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps()
    elif "h5" in dataset or "hdf5" in dataset:
        return load_hdf5(dataset, subset_key)
    else:
        raise ValueError('Not defined for loading %s' % dataset)


def load_data(dataset, subset_key=None):
    x, y = load_data_conv(dataset, subset_key)
    return x.reshape([x.shape[0], -1]), y

if __name__ == "__main__":
    # data = "/home/maximl/DATA/Experiment_data/9-color/earlyfix_d34.h5"
    data = "fmnist"

    ds, _ = load_data_conv(data)

    print(ds[0, :, :, 0].min(),ds[0, :, :, 0].max(), ds[0, :, :, 0].mean())

    from matplotlib import pyplot as plt
    plt.imshow(ds[0, :, :, 0])
    plt.savefig("temp.png")
