/*-----------
 *
 * structs.h
 *
 * This is the head file for structs in this benchmark.
 *
 * streamsOptBenchmark/structs.h
 *
 * By Hao Li
 *
 *------------
 */

#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdio.h>
#include <cuda_runtime.h>

#include <helper_cuda.h> /// -I Developer/NVIDIA/CUDA-7.5/samples/common/inc/helper_cuda.h

cudaError_t error;

#define GEN_NUMBER_RANGE	10000
#define MATRIX_SIZE 		25 // a MATRIX_SIZE*MATRIX_SIZE matrix
#define MATRIX_ELMENT_NUM	(MATRIX_SIZE * MATRIX_SIZE)
#define DATA_SIZE 			(MATRIX_ELMENT_NUM * 2)

#define DEFAULT_THREADS		1024

#define TOTAL_KERNEL_NUM	13

// To decide allocate memory on CPU or on GPU
#define onHOST		1
#define onDEVICE	0

// To choose if using default kernel parameters
#define DEFAULT		0
#define MANUAL		1
#define RANDOM		2
#define OPTMIZE		3

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct matrix{
    int width;
    int height;
    int stride;
    // float element[MATRIX_ELMENT_NUM];
    float *element;
} Matrix;

typedef struct kernel_info
{
	// char* 	name;
	int 	reg;		// register number
	int 	sharedMem;	// shared memory
	int 	dataSize;
	dim3 	BlockDim;	// threads
	dim3 	GridDim;	// block number
	// int 	loopTimes;	//= 1;
} Kernel_info;

static int MaxDataSize = 0;

#endif /* STRUCTS_H */
