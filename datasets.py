import numpy as np
import tensorflow as tf
import time
import h5py

class DatasetWrapper:
    def __init__(self, h5fp, subset_key=None):
        channels = [k for k in list(h5fp.keys()) if "channel" in k]
        images_shape = list(h5fp[channels[0]]["images"].shape)
        ims = np.empty(shape=images_shape, dtype=np.float32)
        masks = np.empty(shape=images_shape, dtype=np.int8)

        if subset_key is not None:
            indices = h5fp["subsets"][subset_key][:]
            images_shape[0] = len(indices)
        else:
            indices = np.s_[:]

        shape = tuple([len(channels)] + images_shape)
        self.images = np.empty(shape=shape, dtype=np.float32)
        
        for i, chan in enumerate(channels):
            h5fp[chan]["masks"].read_direct(ims)
            h5fp[chan]["masks"].read_direct(masks)

            self.images[i] = np.multiply(ims[indices], masks[indices], dtype=np.float32)
            self.images[i] = (self.images[i]-self.images[i].min())/(self.images[i].max()+self.images[i].min())
        self.images = np.moveaxis(self.images, 0, -1)


def load_hdf5(dataset, subset_key=None):
    with h5py.File(dataset) as h5fp:  
        ds = DatasetWrapper(
            h5fp,
            subset_key
        )
    
    print("HDF5 samples", ds.images.shape)
    return ds.images, None


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
    data = "/home/maximl/DATA/Experiment_data/9-color/earlyfix_d34.h5"

    ds, _ = load_data_conv(data)

    print(ds[0, :, :, 0].min(),ds[0, :, :, 0].max())

    from matplotlib import pyplot as plt
    plt.imshow(ds[0, :, :, 0])
    plt.savefig("temp.png")
