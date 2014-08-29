import numpy as N
import numpy
import os
import ctypes

_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('convolution', _path)
lib.conv2d.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                            N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                            N.ctypeslib.ndpointer(N.float32, flags='aligned'),
                            N.ctypeslib.ndpointer(N.float32, flags='aligned')]

def conv2d(data, filters, biases):
    dataSize = int(numpy.sqrt(data.shape[0]))
    filterSize = int(numpy.sqrt(filters.shape[0]))
    nFilters = filters.shape[1]

    data = N.require(data, numpy.float32, ['ALIGNED'])
    filters = N.require(filters, numpy.float32, ['ALIGNED'])
    
    resultSize = dataSize - filterSize + 1;

    result = numpy.zeros((nFilters, resultSize, resultSize), dtype = N.float32)
    lib.conv2d(data, dataSize, filters, filterSize, nFilters, biases, result);
    return result
