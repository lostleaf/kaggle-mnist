import mxnet as mx
import numpy as np
import logging
import matplotlib.pyplot as plt

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

    net = mx.sym.FullyConnected(data=net, num_hidden=10)
    net = mx.sym.SoftmaxOutput(data=net, name="softmax")
    return net

def main():
    train_dataset = np.load("train.npz")
    train_data = train_dataset["train_data"]
    train_labels = train_dataset["train_labels"]
    train_data = train_data.reshape((-1, 1, 28, 28))

    p = np.random.permutation(train_data.shape[0])
    train_data = train_data[p]
    train_labels = train_labels[p]

    net = symbol_lenet()
    print net.list_arguments()
    
    train_iter = mx.io.NDArrayIter(train_data[:40000], train_labels[:40000], batch_size=BATCH_SIZE)
    val_iter = mx.io.NDArrayIter(train_data[40000:], train_labels[40000:], batch_size=BATCH_SIZE)

    model = mx.model.FeedForward(ctx=mx.gpu(0), 
                                 symbol=net, 
                                 num_epoch=200, 
                                 learning_rate=1e-4, 
                                 momentum=0.9, 
                                 wd=1e-5)
    
    model.fit(X=train_iter,
              eval_data=val_iter,
              batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 625))

if __name__ == "__main__":
    main()
