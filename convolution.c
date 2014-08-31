#include <stdio.h>

float inline sigmoid(float x) {
    return (x >= 0) ? x : 0;
}

void conv2d(float *data, int dataSize, 
            float *filters, int filterSize, int nFilters,
            float *biases,
            float *output)
{
    int iFilter, iRow, iCol, iFilterRow, iFilterCol;
    int outputSize = dataSize + 1 - filterSize;
    float value;

    #pragma omp parallel for default(shared) private(iFilter,iRow, iCol, iFilterRow, iFilterCol, value)
    for (iFilter = 0; iFilter < nFilters; iFilter++) {
            for (iRow = 0; iRow <  outputSize; iRow++) {
            for (iCol = 0; iCol < outputSize; iCol++) {
                //this is each of the outputs of the convolution. Each pixel in each output channel
                value = biases[iFilter];
                for (iFilterRow = 0; iFilterRow < filterSize; iFilterRow++) {
                    for (iFilterCol = 0; iFilterCol < filterSize; iFilterCol++) {
                        value += data[(iRow + iFilterRow) * dataSize + iCol + iFilterCol ] * filters[(iFilterRow * filterSize + iFilterCol) * (nFilters) + iFilter];
                    }
                }
                output[iFilter * (outputSize * outputSize) + iRow * outputSize + iCol] = sigmoid(value);
            }
        }
    }
}

void pool2d(float *data, int dataSize, int nChannels,
            int poolSize, float *output) {
    int iChannel, iRow, iCol, iPoolRow, iPoolCol;

    int outputSize = dataSize + 1 - poolSize;
    float maxValue, thisValue;

    for (iChannel = 0; iChannel < nChannels; iChannel++) {
        for (iRow = 0; iRow <  outputSize; iRow++) {
            for (iCol = 0; iCol < outputSize; iCol++) {
                maxValue = 0;
                for (iPoolRow = 0; iPoolRow < poolSize; iPoolRow++) {
                    for (iPoolCol = 0; iPoolCol < poolSize; iPoolCol++) {
                        // data[iChannel][iRow + iPoolRow][iCol + iPoolCol]
                        thisValue = data[iChannel * (dataSize * dataSize) + (iRow + iPoolRow) * dataSize + iCol + iPoolCol];
                        if (thisValue > maxValue)
                            maxValue = thisValue;
                    }
                }
                output[iChannel * (outputSize * outputSize) + iRow * outputSize + iCol] = maxValue;
            }
        }
    }
}
            
