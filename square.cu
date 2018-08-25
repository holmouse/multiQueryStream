/*-----------
 *
 * square.cu
 *
 * This is the source file of an increment kernel.
 *
 * This kernel is from CUDA samples. simpleOccupancy.cu
 *
 * streamsOptBenchmark/square.cu
 *
 * By Hao Li
 *
 *------------
 */

 // #include "functions.h"

////////////////////////////////////////////////////////////////////////////////
// Test kernel
//
// This kernel squares each array element. Each thread addresses
// himself with threadIdx and blockIdx, so that it can handle any
// execution configuration, including anything the launch configurator
// API suggests.
////////////////////////////////////////////////////////////////////////////////
__global__ void square(float *in_array, float *out_array, int arrayCount)
{
	for(int l = 0; l < 1000000; l++)
    {
    // extern __shared__ int dynamicSmem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    out_array[idx] = in_array[idx];
    if (idx < arrayCount) {
        out_array[idx] *= out_array[idx];
    }
	}
}

// int main(int argc, char **argc){
// 	int matrixDataSize = MATRIX_SIZE * MATRIX_SIZE;

// 	printf("%d\n", matrixDataSize);

// 	Matrix h_A, d_A;

// 	initMatrix(h_A, matrixDataSize, onHOST);
// 	initMatrix(d_A, matrixDataSize, onDEVICE);

// 	cudaMemcpy(d_A.elements, h_A.elements, matrixDataSize, cudaMemcpyHostToDevice);

// 	// set kernel launch configuration
//     dim3 threads = dim3(512, 1);
//     dim3 blocks  = dim3(matrixDataSize / threads.x, 1);

//     square<<<blocks, threads, 0, 0>>>(d_A.elements, 100000);

//     cudaMemcpy(h_A.elements, d_A.elements, matrixDataSize, cudaMemcpyDeviceToHost);

//     free(h_A.elements);
//     cudaFree(d_A.elements);

// 	return 0;
// }
