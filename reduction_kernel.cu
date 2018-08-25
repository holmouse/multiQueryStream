/*-----------
 *
 * matrixMulGlobal.cu
 *
 * This is the source file for matrix multiplication with global memory only.
 *
 * This kernel is from NVIDIA CUDA samples. reduction_kernel.cu.
 *
 * streamsOptBenchmark/reduction_kernel.cu
 *
 * By Hao Li
 *
 *------------
 */

/*
    Parallel reduction kernels
*/

#include <stdio.h>
// #include "structs.h"
// #include "functions.h"

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
struct SharedMemory
{
    __device__ inline operator       float *()
    {
        extern __shared__ float __smem[];
        return (float *)__smem;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ float __smem[];
        return (float *)__smem;
    }
};

// // specialize for double to avoid unaligned memory
// // access compile errors
// template<>
// struct SharedMemory<double>
// {
//     __device__ inline operator       double *()
//     {
//         extern __shared__ double __smem_d[];
//         return (double *)__smem_d;
//     }

//     __device__ inline operator const double *() const
//     {
//         extern __shared__ double __smem_d[];
//         return (double *)__smem_d;
//     }
// };

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
__global__ void reduce0(float *g_idata, float *g_odata, unsigned int n)
{
    for(int l = 0; l < 100000; l++)
    {
    float *sdata = SharedMemory();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
__global__ void reduce1(float *g_idata, float *g_odata, unsigned int n)
{
    for(int l = 0; l < 100000; l++)
    {
    float *sdata = SharedMemory();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
__global__ void reduce2(float *g_idata, float *g_odata, unsigned int n)
{
    for(int l = 0; l < 100000; l++)
    {
    float *sdata = SharedMemory();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
__global__ void reduce3(float *g_idata, float *g_odata, unsigned int n)
{
    for(int l = 0; l < 100000; l++)
    {
    float *sdata = SharedMemory();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    int mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        mySum += g_idata[i+blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
// void reduce(int size, int threads, int blocks,
//        int whichKernel, int *d_idata, int *d_odata)
// {
//     dim3 dimBlock(threads, 1, 1);
//     dim3 dimGrid(blocks, 1, 1);

//     // when there is only one warp per block, we need to allocate two warps
//     // worth of shared memory so that we don't index shared memory out of bounds
//     int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

//     // choose which of the optimized versions of reduction to launch
//     switch (whichKernel)
//     {
//         case 0:
//             reduce0<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
//             break;

//         case 1:
//             reduce1<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
//             break;

//         case 2:
//             reduce2<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
//             break;

//         case 3:
//             reduce3<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
//             break;
//     }
// }

// int main(int argc, char **argv){
//  int matrixDataSize = sizeof(int) * MATRIX_SIZE * MATRIX_SIZE;

//  Matrix h_A, h_C;
//  Matrix d_A, d_C;

//  initMatrix(h_A, matrixDataSize, onHOST);
//  initMatrix(h_C, matrixDataSize, onHOST);
//  initMatrix(d_A, matrixDataSize, onDEVICE);
//  initMatrix(d_C, matrixDataSize, onDEVICE);

//  cudaMemcpy(d_A.elements, h_A.elements, matrixDataSize, cudaMemcpyHostToDevice);

//  // Invoke kernel
//  // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//  // dim3 dimGrid(h_B.width / dimBlock.x, h_A.height / dimBlock.y);

//  // execute the kernel
//  for(int i =0; i < 4; i++){
//  reduce(matrixDataSize, h_A.width / BLOCK_SIZE * h_A.height / BLOCK_SIZE, 
//     BLOCK_SIZE*BLOCK_SIZE, i, d_A.elements, d_C.elements);
// }

//  cudaMemcpy(h_C.elements, d_C.elements, matrixDataSize, cudaMemcpyDeviceToHost);

//  free(h_A.elements);
//  free(h_C.elements);
//  cudaFree(d_A.elements);
//  cudaFree(d_C.elements);

//  return 0;
// }
