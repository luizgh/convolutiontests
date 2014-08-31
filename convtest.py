import numpy
import numpy as np
#from scipy.misc import logsumexp
import pyconvolution

def logsumexp(x):
    xmax = x.max()
    return xmax + numpy.log(numpy.sum(numpy.exp(x-xmax)))

#y = y.squeeze().astype(int)

def sigma(x):
    return max(x,0)

class CNNModel:
    def __init__(self):
        self.conv_w = numpy.load("conv1_w.npy")
        self.conv_b = numpy.load("conv1_b.npy")
        self.fc_w = numpy.load("fc10_w.npy")
        self.fc_b = numpy.load("fc10_b.npy")


    def fprop(self,data_x):
        nChannels, outputSize, filterSize = (64,46, 3)
        poolSize = 3

        self.conv_r = pyconvolution.conv2d(data_x, self.conv_w, self.conv_b)

        self.poolr = pyconvolution.pool2d(self.conv_r, 3)

        self.fc_input = self.poolr.reshape(-1)

        self.result = numpy.dot(self.fc_w.T, self.fc_input) + self.fc_b
        self.probs_r = numpy.exp(self.result - logsumexp(self.result))

        return self.probs_r

    def bProp(self, y_expanded):
        hx = self.probs_r
        m = hx.shape[0]

        dLastLayer = hx - y_expanded
        W_grad = (1./m) * numpy.dot(self.fc_input.reshape(-1,1), dLastLayer)
        B_grad = (1./m) * np.sum(dLastLayer, axis=0)
        return W_grad, B_grad

def CrossEntropyError(probs, y_expanded):
    return -numpy.sum(y_expanded * numpy.log(probs)) / probs.shape[0]

