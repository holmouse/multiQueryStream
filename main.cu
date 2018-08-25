/*-----------
 *
 * main.cu
 *
 * This is the source file for main function of this benckmark.
 *
 * streamsOptBenchmark/main.cu
 *
 * By Hao Li
 *
 * Command Line Argument:
 *
 * no prefix: running all the kernels with default block size and block number
 * "--help",			show user help
 * "--manual [num]",	running random number of kernels and input each kernel's informantion manually.
 * "--benchmark [num]", running random number of kernels, got the results of both assigning kernel
 *						info ramdom and optimized.
 *
 *------------
 */

#include "structs.h"
#include "functions.cuh"

#define NUM_BLOCKS    64
#define NUM_THREADS   256


int main(int argc, char **argv){

	primitiveFunc(argc, argv);

	return 0;




	// int matrixDataSize = MATRIX_SIZE * MATRIX_SIZE;

	// printf("%d\n", matrixDataSize);

	// Matrix h_A, d_A;

	// initMatrix(h_A, matrixDataSize, onHOST);
	// initMatrix(d_A, matrixDataSize, onDEVICE);

	// cudaMemcpy(d_A.elements, h_A.elements, matrixDataSize, cudaMemcpyHostToDevice);

	// // set kernel launch configuration
 //    dim3 threads = dim3(512, 1);
 //    dim3 blocks  = dim3(matrixDataSize / threads.x, 1);


 //    Kernel_info kernel[TOTAL_KERNEL_NUM];
 //    initKernel(kernel, DEFAULT, matrixDataSize);

 //    for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
	// 		printf("%d\t", kernel[i].BlockDim.x);
	// 		printf("%d\n", kernel[i].GridDim.x);
	// }

 //    // increment_kernel<<<blocks, threads, 0, 0>>>(d_A.elements, 100000);
 //    // square<<<blocks, threads, 0, 0>>>(d_A.elements, 100000);

 //    cudaMemcpy(h_A.elements, d_A.elements, matrixDataSize, cudaMemcpyDeviceToHost);

 //    free(h_A.elements);
 //    cudaFree(d_A.elements);

	// return 0;
}



//  int main(int argc, char **argv)
// {
//     printf("CUDA Clock sample\n");

//     float *dinput = NULL;
//     float *doutput = NULL;
//     clock_t *dtimer = NULL;

//     clock_t timer[NUM_BLOCKS * 2];
//     float input[NUM_THREADS * 2];

//     for (int i = 0; i < NUM_THREADS * 2; i++)
//     {
//         input[i] = (float)i;
//     }

//     cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2);
//     cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS);
//     cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2);

//     cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice);

//     timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 *NUM_THREADS>>>(dinput, doutput, dtimer);

//     cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost);

//     cudaFree(dinput);
//     cudaFree(doutput);
//     cudaFree(dtimer);


//     // Compute the difference between the last block end and the first block start.
//     clock_t minStart = timer[0];
//     clock_t maxEnd = timer[NUM_BLOCKS];

//     for (int i = 1; i < NUM_BLOCKS; i++)
//     {
//         minStart = timer[i] < minStart ? timer[i] : minStart;
//         maxEnd = timer[NUM_BLOCKS+i] > maxEnd ? timer[NUM_BLOCKS+i] : maxEnd;
//     }

//     printf("Total clocks = %d\n", (int)(maxEnd - minStart));


//     // cudaDeviceReset causes the driver to clean up all state. While
//     // not mandatory in normal operation, it is good practice.  It is also
//     // needed to ensure correct operation when the application is being
//     // profiled. Calling cudaDeviceReset causes all profile data to be
//     // flushed before the application exits
//     cudaDeviceReset();

//     return EXIT_SUCCESS;
// }