import numpy
from scipy.misc import logsumexp

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

        self.conv_r = numpy.zeros((nChannels, outputSize,outputSize), dtype=numpy.float)
        for k in range(nChannels):
            for i in range(outputSize):
                for j in range(outputSize):
                    self.conv_r[k,i,j] = sigma(numpy.sum(data_x.reshape(48,48)[i:i+filterSize,j:j+filterSize] * self.conv_w[:,k].reshape(filterSize,filterSize)) + self.conv_b[k,0])

        self.poolr = numpy.zeros((64, 44,44), dtype=numpy.float)
        for k in range(64):
            for i in range(44):
                for j in range(44):
                    self.poolr[k,i,j] = numpy.max(self.conv_r[k,i:i+poolSize,j:j+poolSize])

        fc_input = self.poolr.reshape(-1)

        self.result = numpy.dot(self.fc_w.T, fc_input) + self.fc_b
        self.probs_r = numpy.exp(self.result - logsumexp(self.result))

        return self.probs_r

    def bProp(self, y):
        hx = self.probs_r

def CrossEntropyError(probs, y):
    y_expanded = numpy.zeros(probs.shape)
    y_expanded[:,y] = 1
    return -numpy.sum(y_expanded * numpy.log(probs))

