import mxnet as mx
import numpy as np
import logging
import matplotlib.pyplot as plt

from itertools import izip

BATCH_SIZE = 64
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def symbol_lenet():
    net = mx.sym.Variable("data")

    net = mx.sym.Convolution(data=net, kernel=(3, 3), pad=(1, 1), num_filter=32)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, pool_type="max", kernel=(2, 2), stride=(2, 2))

    net = mx.sym.Convolution(data=net, kernel=(3, 3), pad=(1, 1), num_filter=64)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, pool_type="max", kernel=(2, 2), stride=(2, 2))

    net = mx.sym.Flatten(data=net)

    net = mx.sym.FullyConnected(data=net, num_hidden=512)
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Dropout(data=net, p=0.5)

    net = mx.sym.FullyConnected(data=net, num_hidden=10)
    net = mx.sym.SoftmaxOutput(data=net, name="softmax")
    return net

def get_data():
    train_dataset = np.load("train.npz")
    train_data = train_dataset["train_data"]
    train_labels = train_dataset["train_labels"]
    train_data = train_data.reshape((-1, 1, 28, 28))
    test_data = np.load("test.npy")
    test_data = test_data.reshape((-1, 1, 28, 28))
    return train_data, train_labels, test_data

def init(arg_dict):
    uniform_init = mx.init.Uniform(scale=0.01)
    for name, arg in arg_dict.iteritems():
        if "weight" in name:
            uniform_init(name, arg)
        if "bias" in name:
            arg[:] = 0

def main():
    train_data, train_labels, test_data = get_data()

    p = np.random.permutation(train_data.shape[0])
    train_data = train_data[p]
    train_labels = train_labels[p]

    train_iter = mx.io.NDArrayIter(train_data[:40000], train_labels[:40000], batch_size=BATCH_SIZE)
    val_iter = mx.io.NDArrayIter(train_data[40000:], train_labels[40000:], batch_size=BATCH_SIZE)

    net = symbol_lenet()
    data_shape = (BATCH_SIZE, 1, 28, 28)
    exe = net.simple_bind(ctx=mx.cpu(0), data=data_shape)

    init(exe.arg_dict)

    optimizer = mx.optimizer.Adam(learning_rate=1e-4)
    updater = mx.optimizer.get_updater(optimizer)

    accuracy = mx.metric.Accuracy()

    data = exe.arg_dict["data"]
    label = exe.arg_dict["softmax_label"]

    for epoch in xrange(100):
        print "Epoch:", epoch

        train_iter.reset()
        accuracy.reset()

        for idx_iter, batch in enumerate(train_iter):
            data[:] = batch.data[0]
            label[:] = batch.label[0]

            exe.forward(is_train=True)
            exe.backward()

            for idx, (weight, grad) in enumerate(izip(exe.arg_arrays, exe.grad_arrays)):
                updater(idx, grad, weight)

            accuracy.update(batch.label, exe.outputs)
            if idx_iter % 10 == 0:
                print "Iteration: ", idx_iter, "Train accuracy: ", accuracy.get()[1]

        val_iter.reset()
        accuracy.reset()

        for idx_iter, batch in enumerate(val_iter):
            data[:] = batch.data[0]
            label[:] = batch.label[0]

            exe.forward(is_train=False)

if __name__ == "__main__":
    main()
