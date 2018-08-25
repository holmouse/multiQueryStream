/*-----------
 *
 * atomics.cu
 *
 * This is the source file of antomic operations.
 *
 * This kernel is based on CUDA samples. simpleAtomicIntrinsics.cuh
 *
 * streamsOptBenchmark/atomics.cu
 *
 * By Hao Li
 *
 *------------
 */

#include <time.h>
#include <cuda_runtime.h>

// #include "functions.cuh"

 __global__ void atomicFunc(float *g_idata, float *g_odata)
{
    for(int l = 0; l < 100000; l++)
    {

    // access thread id
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Test various atomic instructions

    // Arithmetic atomic instructions

    int i = 0;

    while(g_odata[i] != NULL)
    {
        g_odata[i] = g_idata[i];
        // Atomic addition
        atomicAdd(&g_odata[i], 10.0);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic subtraction (final should be 0)
        atomicSub((int *)&g_odata[i], 10);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic exchange
        atomicExch(&g_odata[i], (float)tid);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic maximum
        atomicMax((int *)&g_odata[i], tid);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic minimum
        atomicMin((int *)&g_odata[i], tid);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic increment (modulo 17+1)
        atomicInc((unsigned int *)&g_odata[i], 17);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic decrement
        atomicDec((unsigned int *)&g_odata[i], 137);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic compare-and-swap
        atomicCAS((int *)&g_odata[i], tid-1, tid);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Bitwise atomic instructions

        // Atomic AND
        atomicAnd((int *)&g_odata[i], 2*tid+7);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic OR
        atomicOr((int *)&g_odata[i], 1 << tid);

        if(g_odata[i++] != NULL)
            break;

        g_odata[i] = g_idata[i];
        // Atomic XOR
        atomicXor((int *)&g_odata[i], tid);
        i++;
    }

    }
}

// int main(int argc, char **argv)
// {
//     unsigned int numThreads = 256;
//     unsigned int numBlocks = 64;
//     unsigned int numData = 1000000;
//     unsigned int memSize = sizeof(int) * numData;

//     //allocate mem for the result on host side
//     int *hOData = (int *) malloc(memSize);

//     //initalize the memory
//     for (unsigned int i = 0; i < numData; i++)
//         hOData[i] = 0;

//     //To make the AND and XOR tests generate something other than 0...
//     hOData[8] = hOData[10] = 0xff;

//     // allocate device memory for result
//     float *dOData;
//     cudaMalloc((void **) &dOData, sizeof(float) * memSize);
//     // copy host memory to device to initialize to zers
//     cudaMemcpy(dOData, hOData, sizeof(float) * memSize, cudaMemcpyHostToDevice);

//     cudaEvent_t start;
//     error = cudaEventCreate(&start);

//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

//     cudaEvent_t stop;
//     error = cudaEventCreate(&stop);

//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

//     // Record the start event
//     error = cudaEventRecord(start, NULL);

//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

//     // execute the kernel
//     atomicFunc<<<numBlocks, numThreads>>>(dOData);

//     // Record the stop event
//     error = cudaEventRecord(stop, NULL);

//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

//     // Wait for the stop event to complete
//     error = cudaEventSynchronize(stop);

//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

//     float msecTotal = 0.0f;
//     error = cudaEventElapsedTime(&msecTotal, start, stop);

//     printf("Running Time: %f ms\n", msecTotal);

//     cudaMemcpy(hOData, dOData, memSize, cudaMemcpyDeviceToHost);

//     free(hOData);
//     cudaFree(dOData);

//     return 0;
// }
