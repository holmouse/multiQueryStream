/*-----------
 *
 * matrixMulGlobal.cu
 *
 * This is the source file for matrix multiplication with global memory only.
 *
 * This kernel is modified based on NVIDIA CUDA_C_Programming_Guide.
 *
 * streamsOptBenchmark/matrixMulGlobal.cu
 *
 * By Hao Li
 *
 *------------
 */

#include "structs.h"
#include "functions.cuh"

__global__ void MatMulGlobal(float *D_data, float *D_result, int MatrixSize)
// __global__ void MatMulGlobal(Matrix A, Matrix B, Matrix C)
{
    for(int l = 0; l < 100000; l++)
    {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / MatrixSize;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) % MatrixSize;

    // int row = (blockIdx.x * blockDim.x + threadIdx.x) / A.width;
    // int col = (blockIdx.x * blockDim.x + threadIdx.x) % B.width;

    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < MatrixSize; ++e)
        Cvalue += D_data[row * MatrixSize + e]
                * D_data[MatrixSize * MatrixSize + e * MatrixSize + col];
    D_result[row * MatrixSize + col] = Cvalue;

    // for (int e = 0; e < A.width; ++e)
    //     Cvalue += A.element[row * A.width + e]
    //             * B.element[e * B.width + col];
    // C.element[row * C.width + col] = Cvalue;

    // printf("%d: %f\n", row * C_width + col, C_element[row * C_width + col]);
    }
}

// int main(int argc, char **argv){
// 	float *h_data, *h_result;
// 	h_data = (float *)malloc(sizeof(float) * DATA_SIZE);
// 	h_result = (float *)malloc(sizeof(float) * DATA_SIZE);
// 	// float data[DATA_SIZE];
// 	constantInit(h_data, DATA_SIZE);

// 	float *d_data, *d_result;

// 	error = cudaMalloc((void **) &d_data, sizeof(float) * DATA_SIZE);

// 	if (error != cudaSuccess)
//     {
//        	printf("cudaMalloc (d_data) returned error code %d, line(%d)\n", error, __LINE__);
//        	exit(EXIT_FAILURE);
//    	}

//    	error = cudaMalloc((void **) &d_result, sizeof(float) * DATA_SIZE);

// 	if (error != cudaSuccess)
//     {
//        	printf("cudaMalloc (d_result) returned error code %d, line(%d)\n", error, __LINE__);
//        	exit(EXIT_FAILURE);
//    	}

// 	// Matrix h_A, h_B, h_C;
// 	// Matrix d_A, d_B, d_C;

// 	// initMatrix(h_A, data, 0, onHOST);
// 	// initMatrix(h_B, data, MATRIX_ELMENT_NUM, onHOST);
// 	// initMatrix(h_C, data, 0, onHOST);
// 	// initMatrix(d_A, data, 0, onDEVICE);
// 	// initMatrix(d_B, data, 0, onDEVICE);
// 	// initMatrix(d_C, data, 0, onDEVICE);

// 	error = cudaMemcpy(d_data, h_data, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice);

// 	if (error != cudaSuccess)
//     {
//         printf("cudaMemcpy (d_data,h_data) returned error code %d, line(%d)\n", error, __LINE__);
//         exit(EXIT_FAILURE);
//     }

// 	cudaEvent_t start;
// 	error = cudaEventCreate(&start);

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

// 	// Invoke kernel
// 	dim3 dimBlock(256, 1);
// 	dim3 dimGrid(ceil(DATA_SIZE / 256), 1);

// 	// Record the start event
//     error = cudaEventRecord(start, NULL);

//     if (error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//     }

// 	MatMulGlobal<<<dimGrid, dimBlock>>>(d_data, d_result, MATRIX_SIZE);
// 	// MatMulGlobal<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

// 	// Record the stop event
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

//     printf("time: %f\n", msecTotal);

// 	error = cudaMemcpy(h_result, d_result, sizeof(float) * MATRIX_ELMENT_NUM, cudaMemcpyDeviceToHost);

// 	if (error != cudaSuccess)
//     {
//         printf("cudaMemcpy (h_result,d_result) returned error code %d, line(%d)\n", error, __LINE__);
//         exit(EXIT_FAILURE);
//     }

// 	// for(int i = 0; i < MATRIX_ELMENT_NUM; i++){
// 	// 	printf("%d: %f\n", i, h_result[i]);
// 	// }

// 	free(h_data);
// 	free(h_result);
// 	cudaFree(d_data);
// 	cudaFree(d_result);

// 	return 0;
// }
