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

    for (iFilter = 0; iFilter < nFilters; iFilter++) {
            #pragma omp parallel for default(shared) firstprivate(iFilter) private(iRow, iCol, iFilterRow, iFilterCol)
            for (iRow = 0; iRow <  outputSize; iRow++) {
            for (iCol = 0; iCol < outputSize; iCol++) {
                //this is each of the outputs of the convolution. Each pixel in each output channel
                float value = biases[iFilter];
                /*
                if (iFilter == 10 && iRow ==20 && iCol==9)
                    printf("Bias for filter %d: %f\n ", iFilter, value);
                    */
                for (iFilterRow = 0; iFilterRow < filterSize; iFilterRow++) {
                    for (iFilterCol = 0; iFilterCol < filterSize; iFilterCol++) {
                        value += data[(iRow + iFilterRow) * dataSize + iCol + iFilterCol ] * filters[(iFilterRow * filterSize + iFilterCol) * (nFilters) + iFilter];
                        /*
                        if (iFilter == 10 && iRow ==20 && iCol==9) {
                        printf("Filter %d at %d, %d: %f. Image at %d, %d: %f\n", iFilter, iFilterRow, iFilterCol, filters[(iFilterRow * filterSize + iFilterCol) * (nFilters) + iFilter],
                                iRow + iFilterRow, iCol + iFilterCol, data[(iRow + iFilterRow) * dataSize + iCol + iFilterCol ]);
                        }*/
                    }
                }/*
                if (iFilter == 10 && iRow ==20 && iCol==9)
                    printf("Output for filter %d, row %d, col %d: %f (%f)", iFilter, iRow, iCol, sigmoid(value),value);*/
                output[iFilter * (outputSize * outputSize) + iRow * outputSize + iCol] = sigmoid(value);
            }
        }
    }
}

