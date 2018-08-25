/*-----------
 *
 * matrixMulShared.cu
 *
 * This is the source file for matrix multiplication with global memory only.
 *
 * This kernel is from NVIDIA CUDA_C_Programming_Guide.
 *
 * streamsOptBenchmark/matrixMulShared.cu
 *
 * By Hao Li
 *
 *------------
 */

#include "structs.h"
#include "functions.cuh"

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatMul()
 // __global__ void MatMulShared(Matrix A, Matrix B, Matrix C)
 __global__ void MatMulShared(float *D_data, float *D_result, int MatrixSize)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

//  int main(int argc, char **argv){
// 	int matrixDataSize = sizeof(int) * MATRIX_SIZE * MATRIX_SIZE;

// 	Matrix h_A, h_B, h_C;
// 	Matrix d_A, d_B, d_C;

// 	initMatrix(h_A, matrixDataSize, onHOST);
// 	initMatrix(h_B, matrixDataSize, onHOST);
// 	initMatrix(h_C, matrixDataSize, onHOST);
// 	initMatrix(d_A, matrixDataSize, onDEVICE);
// 	initMatrix(d_B, matrixDataSize, onDEVICE);
// 	initMatrix(d_C, matrixDataSize, onDEVICE);

// 	cudaMemcpy(d_A.elements, h_A.elements, matrixDataSize, cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_B.elements, h_B.elements, matrixDataSize, cudaMemcpyHostToDevice);

// 	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// 	dim3 dimGrid(h_B.width / dimBlock.x, h_A.height / dimBlock.y);
// 	MatMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

// 	cudaMemcpy(h_C.elements, h_C.elements, matrixDataSize, cudaMemcpyDeviceToHost);

// 	free(h_A.elements);
// 	free(h_B.elements);
// 	free(h_C.elements);
// 	cudaFree(d_A.elements);
// 	cudaFree(d_B.elements);
// 	cudaFree(d_C.elements);

// 	return 0;
// }