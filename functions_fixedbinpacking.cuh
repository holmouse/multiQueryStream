/*-----------
 *
 * functions.cuh
 *
 * This is the head file of non-cuda functions in this benchmark.
 *
 * streamsOptBenchmark/functions.cuh
 *
 * By Hao Li
 *
 *------------
 */

#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH

#include "structs.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Include all the cuda kernels
#include "atomics.cu"
#include "clock.cu"
#include "distanceGlobal.cu"
#include "distanceShared.cu"
#include "increment_kernel.cu"
#include "layerTransform.cu"
#include "matrixMulGlobal.cu"
// #include "matrixMulShared.cu"
#include "reduction_kernel.cu"
#include "square.cu"
#include "vectorAdd.cu"
#include "manualKernels.cu"

// Ceiling for integers, e.g. a = 7, b = 3, result is 3
int intCeil(int a, int b){
  if( (a % b) == 0 ){
    return (a / b);
  }else{
    return (a / b + 1);
  }
}

// Initialize data of certain size
extern void constantInit(float *Data, int DataSize)
{
	// *Data = (float *)malloc(sizeof(float) * DATA_SIZE);
	time_t t;
	srand((unsigned) time(&t));
    for (int i = 0; i < DataSize; ++i)
    {
        // data[i] = rand()%(size+1);	// random set data[i] from 0 to size
        Data[i] = ((float)rand()/(float)(RAND_MAX)) * GEN_NUMBER_RANGE;
        // printf("%d: %f\n", i, Data[i]);
    }
    // printf("done\n");
}

// Initialize a matrix
extern void initMatrix(Matrix &M, float *Data, int DataStartIndex, bool Host)
{
	M.width = MATRIX_SIZE;
	M.height = MATRIX_SIZE;
	M.stride = M.width;

	// // if the matrix is on host, using malloc, else using cudaMalloc
	// if(Host == 1){
	// 	M.element = (float *)malloc(sizeof(float) * MATRIX_ELMENT_NUM);
	// 	for(int i = 0; i < MATRIX_ELMENT_NUM; i++){
	// 		M.element[i] = Data[DataStartIndex + i];
	// 		// printf("%d: %f\n", i, M.element[i]);
	// 	}
	// 	// constantInit(M.element, M.width * M.height);
	// }else{
	// 	error = cudaMalloc((void **) &M.element, sizeof(float) * MATRIX_ELMENT_NUM);

	// 	if (error != cudaSuccess)
 //    	{
 //        	printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
 //        	exit(EXIT_FAILURE);
 //    	}
 //    	printf("done\n");
	// }

	// printf("done\n");
}

// Choose a kernel to launch
extern void lauchKernel(int kernelNum, Kernel_info Kernel, float* D_data, float* D_result, 
	cudaStream_t* Streams, int StreamID)
{

	// cudaEvent_t start;
	// error = cudaEventCreate(&start);
	
	// if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

    // cudaEvent_t stop;
    // error = cudaEventCreate(&stop);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

    // float msecTime = 0.0f;

    // printf("launching kernel %d\n", kernelNum);

	switch(kernelNum)
	{
		case 0:
		{
			// Kernel.GridDim = dim3(ceil(Kernel.dataSize / DEFAULT_THREADS),1,1);

			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// Invoke kernel
			atomicFunc<<<Kernel.GridDim, Kernel.BlockDim, 
				0, Streams[StreamID]>>>(D_data, D_result);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("Atomic Running Time: %f ms\n", msecTime);

    		// return msecTime;

    		break;
    	}
		case 1:
		{
			// Kernel.GridDim = dim3(ceil(Kernel.dataSize / DEFAULT_THREADS),1,1);

			clock_t *dtimer = NULL;

			// clock_t timer[Kernel.BlockDim.x * 2];

			error = cudaMalloc((void **)&dtimer, sizeof(clock_t) * Kernel.BlockDim.x * 2);

			if (error != cudaSuccess)
    		{
       			printf("cudaMalloc (dtimer) returned error code %d, line(%d)\n", error, __LINE__);
       			exit(EXIT_FAILURE);
   			}

   			// // Record the start event
    		// error = cudaEventRecord(start, NULL);

    		// if (error != cudaSuccess)
    		// {
      //   		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
      //   		exit(EXIT_FAILURE);
    		// }

   			// Invoke kernel
   			// clockTimedReduction<<<Kernel.GridDim, Kernel.BlockDim, 
   			// 	sizeof(float) * 2 * Kernel.BlockDim.x, Streams[StreamID]>>>(D_data, D_result, dtimer);
        clockTimedReduction<<<Kernel.GridDim, Kernel.BlockDim, 
          Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result, dtimer);  

   			// // Record the stop event
    		// error = cudaEventRecord(stop, NULL);

    		// if (error != cudaSuccess)
    		// {
      //   		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
      //   		exit(EXIT_FAILURE);
    		// }

    		// // Wait for the stop event to complete
    		// error = cudaEventSynchronize(stop);

    		// if (error != cudaSuccess)
    		// {
      //   		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
      //   		exit(EXIT_FAILURE);
    		// }

    		// msecTime = 0.0f;
    		// error = cudaEventElapsedTime(&msecTime, start, stop);

    		// printf("clockTimedReduction Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 2:
		{

			// // Record the start event
    		// error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

			// Invoke kernel
			gpu_global_distance<<<Kernel.GridDim, Kernel.BlockDim, 
				0, Streams[StreamID]>>>(D_result, D_data, Kernel.dataSize);

			// // Record the stop event
    		// error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
    		// error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// msecTime = 0.0f;
    		// error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("gpu_global_distance Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 3:
		{
			// Kernel.GridDim = dim3(ceil((Kernel.dataSize / 2) / DEFAULT_THREADS),1,1);

			// // Record the start event
    		// error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

			// Invoke kernel
			gpu_shared_memory<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem, Streams[StreamID]>>>(D_result, D_data, (Kernel.dataSize / 2), Kernel.BlockDim.x);

			// // Record the stop event
    		// error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
    		// error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// msecTime = 0.0f;
    		// error = cudaEventElapsedTime(&msecTime, start, stop);

    		// printf("gpu_shared_memory Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 4:
		{
			// // Record the start event
    		// error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// Invoke kernel
			increment_kernel<<<Kernel.GridDim, Kernel.BlockDim, 
				0, Streams[StreamID]>>>(D_data, D_result, 100000);

			// // Record the stop event
    		// error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
    		// error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// msecTime = 0.0f;
    		// error = cudaEventElapsedTime(&msecTime, start, stop);

    		// printf("increment_kernel Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 5:
		{
			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// Invoke kernel
			LayerTransformKernel<<<Kernel.GridDim, Kernel.BlockDim, 
				0, Streams[StreamID]>>>(D_data, D_result, 512, 512, 1);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("LayerTransformKernel Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 6:
		{
			// Kernel.GridDim = dim3(ceil((Kernel.dataSize / 2) / DEFAULT_THREADS),1,1);

			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// Invoke kernel
			MatMulGlobal<<<Kernel.GridDim, Kernel.BlockDim, 
			0 , Streams[StreamID]>>>(D_data, D_result, sqrt(Kernel.dataSize));

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("matrixMulGlobal Running Time: %f ms\n", msecTime);

    		// return msecTime;

    		break;
		}
		// case 7:
		// {
		// 	// Kernel.GridDim = dim3(ceil(Kernel.dataSize / DEFAULT_THREADS),1,1);
		// 	// kernel_7;
		// 	break;
		// }
		case 7:
		{
			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// int smemSize = (Kernel.BlockDim.x <= 32) ? 
    		// 	2 * Kernel.BlockDim.x * sizeof(float) : Kernel.BlockDim.x * sizeof(float);

    		// Invoke kernel
			reduce0<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result, Kernel.dataSize);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("reduce0 Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 8:
		{
			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// int smemSize = (Kernel.BlockDim.x <= 32) ? 
    		// 	2 * Kernel.BlockDim.x * sizeof(float) : Kernel.BlockDim.x * sizeof(float);

    		// Invoke kernel
			reduce1<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result, Kernel.dataSize);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("reduce1 Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 9:
		{
			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// int smemSize = (Kernel.BlockDim.x <= 32) ? 
    		// 	2 * Kernel.BlockDim.x * sizeof(float) : Kernel.BlockDim.x * sizeof(float);

    		// Invoke kernel
			reduce2<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result, Kernel.dataSize);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("reduce2 Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 10:
		{
			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// int smemSize = (Kernel.BlockDim.x <= 32) ? 
    		// 	2 * Kernel.BlockDim.x * sizeof(float) : Kernel.BlockDim.x * sizeof(float);

    		// Invoke kernel
			reduce3<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result, Kernel.dataSize);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("reduce3 Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 11:
		{
			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// Invoke kernel
			square<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem , Streams[StreamID]>>>(D_data, D_result, 100000);

			// // Record the stop event
   //  		error = cudaEventRecord(stop, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		// Wait for the stop event to complete
   //  		error = cudaEventSynchronize(stop);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

   //  		msecTime = 0.0f;
   //  		error = cudaEventElapsedTime(&msecTime, start, stop);

   //  		printf("square Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
		case 12:
		{
			// Kernel.GridDim = dim3(ceil((Kernel.dataSize / 2) / DEFAULT_THREADS),1,1);

			// // Record the start event
   //  		error = cudaEventRecord(start, NULL);

   //  		if (error != cudaSuccess)
   //  		{
   //      		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
   //      		exit(EXIT_FAILURE);
   //  		}

    		// Invoke kernel
			vectorAdd<<<Kernel.GridDim, Kernel.BlockDim, 
				Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result, Kernel.dataSize);

			// Record the stop event
    		// error = cudaEventRecord(stop, NULL);

    		// if (error != cudaSuccess)
    		// {
      //   		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
      //   		exit(EXIT_FAILURE);
    		// }

    		// // Wait for the stop event to complete
    		// error = cudaEventSynchronize(stop);

    		// if (error != cudaSuccess)
    		// {
      //   		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
      //   		exit(EXIT_FAILURE);
    		// }

    		// msecTime = 0.0f;
    		// error = cudaEventElapsedTime(&msecTime, start, stop);

    		// printf("vectorAdd Running Time: %f ms\n", msecTime);

    		// return msecTime;

			break;
		}
	}
	// return msecTime;
}

// Because the kernel running time is too small, we need to 

// Initialize kernels
extern void initKernel(Kernel_info *Kernel, int kernelMode, int Thread, int DataSize){
	time_t t;
  srand((unsigned) time(&t));

  FILE *fp;
  fp = fopen("./threadConf.txt","a+");
  char buffer[10];

  // atomicFunc
	if(kernelMode == DEFAULT)
	{
		Kernel[0].dataSize = DATA_SIZE;
		Kernel[0].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 1 (atomicFunc): ");
		scanf("%d", &Thread);
		Kernel[0].BlockDim = dim3(Thread,1,1);

		printf("Please input total size of data: ");
		scanf("%d", &Kernel[0].dataSize);
		
	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[0].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[0].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[0].BlockDim = dim3(Thread,1,1);
  }

	Kernel[0].reg = 17;
	Kernel[0].GridDim = dim3(intCeil(Kernel[0].dataSize, Kernel[0].BlockDim.x),1,1);
	Kernel[0].sharedMem = 0;
  if(MaxDataSize <= Kernel[0].BlockDim.x * Kernel[0].GridDim.x){
      MaxDataSize = Kernel[0].BlockDim.x * Kernel[0].GridDim.x;
  }
	
	// clockTimedReduction
	if(kernelMode == DEFAULT)
	{
		Kernel[1].dataSize = DATA_SIZE;
		Kernel[1].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 2 (clockTimedReduction): ");
		scanf("%d", &Thread);
		Kernel[1].BlockDim = dim3(Thread,1,1);

		printf("Please input total size of data: ");
		scanf("%d", &Kernel[1].dataSize);

	}else if(kernelMode == RANDOM){ 
    if(DataSize != 0){
      Kernel[1].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[1].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[1].BlockDim = dim3(Thread,1,1);
  }

	Kernel[1].reg = 19;
	Kernel[1].GridDim = dim3(intCeil(Kernel[1].dataSize, Kernel[1].BlockDim.x),1,1);
	//Kernel[1].sharedMem = sizeof(float) * 2 * Kernel[1].BlockDim.x;
	Kernel[1].sharedMem = sizeof(float) * 2 * 1024;
  if(MaxDataSize <= Kernel[1].BlockDim.x * Kernel[1].GridDim.x){
      MaxDataSize = Kernel[1].BlockDim.x * Kernel[1].GridDim.x;
  }

	// gpu_global_distance
	if(kernelMode == DEFAULT)
	{
		Kernel[2].dataSize = DATA_SIZE / 2;
		Kernel[2].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 3 (gpu_global_distance): ");
		scanf("%d", &Thread);
		Kernel[2].BlockDim = dim3(Thread,1,1);

		printf("Please input total number of points: ");
		scanf("%d", &Kernel[2].dataSize);
		// Kernel[2].dataSize /= 2;

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      //Kernel[2].dataSize = DataSize / 2;
	Kernel[2].dataSize = 1250;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[2].dataSize);

    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[2].BlockDim = dim3(Thread,1,1);
  }

	Kernel[2].reg = 17;
	Kernel[2].GridDim = dim3(intCeil(Kernel[2].dataSize, Kernel[2].BlockDim.x),1,1);
	Kernel[2].sharedMem = 0;
  if(MaxDataSize <= Kernel[2].BlockDim.x * Kernel[2].GridDim.x){
      MaxDataSize = Kernel[2].BlockDim.x * Kernel[2].GridDim.x;
  }

	// gpu_shared_memory
	if(kernelMode == DEFAULT)
	{
		Kernel[3].dataSize = DATA_SIZE / 2;
		Kernel[3].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 4 (gpu_shared_memory): ");
		scanf("%d", &Thread);
		Kernel[3].BlockDim = dim3(Thread,1,1);

		printf("Please input total number of points: ");
		scanf("%d", &Kernel[3].dataSize);
		// Kernel[3].dataSize /= 2;

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[3].dataSize = DataSize / 2;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[3].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[3].BlockDim = dim3(Thread,1,1);
  }

	Kernel[3].reg = 25;
	Kernel[3].GridDim = dim3(intCeil(Kernel[3].dataSize, Kernel[3].BlockDim.x),1,1);
	Kernel[3].sharedMem = sizeof(float) * 4 * 1024;	
	//Kernel[3].sharedMem = sizeof(float) * 4 * Kernel[3].BlockDim.x;
  if(MaxDataSize <= Kernel[3].BlockDim.x * Kernel[3].GridDim.x){
      MaxDataSize = Kernel[3].BlockDim.x * Kernel[3].GridDim.x;
  }

	// increment_kernel
	if(kernelMode == DEFAULT)
	{
		Kernel[4].dataSize = DATA_SIZE;
		Kernel[4].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 5 (increment_kernel): ");
		scanf("%d", &Thread);
		Kernel[4].BlockDim = dim3(Thread,1,1);

		printf("Please input total size of data: ");
		scanf("%d", &Kernel[4].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[4].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[4].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[4].BlockDim = dim3(Thread,1,1);
  }

	Kernel[4].reg = 25;
	Kernel[4].GridDim = dim3(intCeil(Kernel[4].dataSize, Kernel[4].BlockDim.x),1,1);
	Kernel[4].sharedMem = 0;
  if(MaxDataSize <= Kernel[4].BlockDim.x * Kernel[4].GridDim.x){
      MaxDataSize = Kernel[4].BlockDim.x * Kernel[4].GridDim.x;
  }	

	// LayerTransformKernel
	if(kernelMode == DEFAULT)
	{
		Kernel[5].dataSize = DATA_SIZE;
		Kernel[5].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 6 (LayerTransformKernel): ");
		scanf("%d", &Thread);
		Kernel[5].BlockDim = dim3(Thread,1,1);

		printf("Please input total size of data: ");
		scanf("%d", &Kernel[5].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[5].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[5].dataSize);

    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[5].BlockDim = dim3(Thread,1,1);
  }

	Kernel[5].reg = 31;
	Kernel[5].GridDim = dim3(intCeil(Kernel[5].dataSize, Kernel[5].BlockDim.x),1,1);
	Kernel[5].sharedMem = 0;
  if(MaxDataSize <= Kernel[5].BlockDim.x * Kernel[5].GridDim.x){
      MaxDataSize = Kernel[5].BlockDim.x * Kernel[5].GridDim.x;
  }

	// MatMulGlobal
	if(kernelMode == DEFAULT)
	{
		Kernel[6].dataSize = DATA_SIZE / 2;
		Kernel[6].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 7 (MatMulGlobal): ");
		scanf("%d", &Thread);
		Kernel[6].BlockDim = dim3(Thread,1,1);

		int matrixSize;
		printf("Please input matrix size: ");
		scanf("%d", &matrixSize);
		Kernel[6].dataSize = matrixSize * matrixSize;

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      //Kernel[6].dataSize = DataSize / 2;
	Kernel[6].dataSize = 2500;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[6].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[6].BlockDim = dim3(Thread,1,1);
  }

	Kernel[6].reg = 20;
	Kernel[6].GridDim = dim3(intCeil(Kernel[6].dataSize, Kernel[6].BlockDim.x),1,1);
	Kernel[6].sharedMem = 0;
  if(MaxDataSize <= Kernel[6].BlockDim.x * Kernel[6].GridDim.x){
      MaxDataSize = Kernel[6].BlockDim.x * Kernel[6].GridDim.x;
  }

	// reduce0
	if(kernelMode == DEFAULT)
	{
		Kernel[7].dataSize = DATA_SIZE;
		Kernel[7].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 8 (reduce0): ");
		scanf("%d", &Thread);
		Kernel[7].BlockDim = dim3(Thread,1,1);

		printf("Please input total number of points: ");
		scanf("%d", &Kernel[7].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[7].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[7].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[7].BlockDim = dim3(Thread,1,1);
  }

	Kernel[7].reg = 14;
	Kernel[7].GridDim = dim3(intCeil(Kernel[7].dataSize, Kernel[7].BlockDim.x),1,1);
	//Kernel[7].sharedMem = (Kernel[7].BlockDim.x <= 32) ? 
    	//		2 * Kernel[7].BlockDim.x * sizeof(float) : Kernel[7].BlockDim.x * sizeof(float);
	Kernel[7].sharedMem = 1024 * 4 * sizeof(float);
  if(MaxDataSize <= Kernel[7].BlockDim.x * Kernel[7].GridDim.x){
      MaxDataSize = Kernel[7].BlockDim.x * Kernel[7].GridDim.x;
  }

	// reduce1
	if(kernelMode == DEFAULT)
	{
		Kernel[8].dataSize = DATA_SIZE;
		Kernel[8].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 9 (reduce1): ");
		scanf("%d", &Thread);
		Kernel[8].BlockDim = dim3(Thread,1,1);

		printf("Please input total number of points: ");
		scanf("%d", &Kernel[8].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[8].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[8].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[8].BlockDim = dim3(Thread,1,1);
  }

	Kernel[8].reg = 14;
	Kernel[8].GridDim = dim3(intCeil(Kernel[8].dataSize, Kernel[8].BlockDim.x),1,1);
	//Kernel[8].sharedMem = (Kernel[8].BlockDim.x <= 32) ? 
    			//2 * Kernel[8].BlockDim.x * sizeof(float) : Kernel[8].BlockDim.x * sizeof(float);
	Kernel[8].sharedMem = 1024 * 6 * sizeof(float);
  if(MaxDataSize <= Kernel[8].BlockDim.x * Kernel[8].GridDim.x){
      MaxDataSize = Kernel[8].BlockDim.x * Kernel[8].GridDim.x;
  }

	// reduce2
	if(kernelMode == DEFAULT)
	{
		Kernel[9].dataSize = DATA_SIZE;
		Kernel[9].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 10 (reduce2): ");
		scanf("%d", &Thread);
		Kernel[9].BlockDim = dim3(Thread,1,1);

		printf("Please input total number of points: ");
		scanf("%d", &Kernel[9].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[9].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[9].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[9].BlockDim = dim3(Thread,1,1);
  }

	Kernel[9].reg = 13;
	Kernel[9].GridDim = dim3(intCeil(Kernel[9].dataSize, Kernel[9].BlockDim.x),1,1);
	//Kernel[9].sharedMem = (Kernel[9].BlockDim.x <= 32) ? 
    	//		2 * Kernel[9].BlockDim.x * sizeof(float) : Kernel[9].BlockDim.x * sizeof(float);
	Kernel[9].sharedMem = 1024 * 8 * sizeof(float);
  if(MaxDataSize <= Kernel[9].BlockDim.x * Kernel[9].GridDim.x){
      MaxDataSize = Kernel[9].BlockDim.x * Kernel[9].GridDim.x;
  }

	// reduce2
	if(kernelMode == DEFAULT)
	{
		Kernel[10].dataSize = DATA_SIZE;
		Kernel[10].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 11 (reduce2): ");
		scanf("%d", &Thread);
		Kernel[10].BlockDim = dim3(Thread,1,1);

		printf("Please input total number of points: ");
		scanf("%d", &Kernel[10].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[10].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[10].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[10].BlockDim = dim3(Thread,1,1);
  }

	Kernel[10].reg = 15;
	Kernel[10].GridDim = dim3(intCeil(Kernel[10].dataSize, Kernel[10].BlockDim.x),1,1);
	//Kernel[10].sharedMem = (Kernel[10].BlockDim.x <= 32) ? 
    	//		2 * Kernel[10].BlockDim.x * sizeof(float) : Kernel[10].BlockDim.x * sizeof(float);
	Kernel[10].sharedMem = 1024 * sizeof(float);
  if(MaxDataSize <= Kernel[10].BlockDim.x * Kernel[10].GridDim.x){
      MaxDataSize = Kernel[10].BlockDim.x * Kernel[10].GridDim.x;
  }

	// square
	if(kernelMode == DEFAULT)
	{
		Kernel[11].dataSize = DATA_SIZE;
		Kernel[11].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 12 (square): ");
		scanf("%d", &Thread);
		Kernel[11].BlockDim = dim3(Thread,1,1);

		printf("Please input total size of data: ");
		scanf("%d", &Kernel[11].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[11].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[11].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[11].BlockDim = dim3(Thread,1,1);
  }

	Kernel[11].reg = 8;
	Kernel[11].GridDim = dim3(intCeil(Kernel[11].dataSize, Kernel[11].BlockDim.x),1,1);
	Kernel[11].sharedMem = sizeof(float) * 10 * 1024;
  if(MaxDataSize <= Kernel[11].BlockDim.x * Kernel[11].GridDim.x){
      MaxDataSize = Kernel[11].BlockDim.x * Kernel[11].GridDim.x;
  }

	// vectorAdd
	if(kernelMode == DEFAULT)
	{
		Kernel[12].dataSize = DATA_SIZE / 2;
		Kernel[12].BlockDim = dim3(Thread,1,1);
	}else if(kernelMode == MANUAL){
		// int thread = 0;
		printf("Please input block size of kernel 13 (vectorAdd): ");
		scanf("%d", &Thread);
		Kernel[12].BlockDim = dim3(Thread,1,1);

		printf("Please input total size of data: ");
		scanf("%d", &Kernel[12].dataSize);

	}else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[12].dataSize = DataSize / 2;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[12].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[12].BlockDim = dim3(Thread,1,1);
  }

	Kernel[12].reg = 12;
	Kernel[12].GridDim = dim3(intCeil(Kernel[12].dataSize, Kernel[12].BlockDim.x),1,1);
	Kernel[12].sharedMem = sizeof(float) * 5 * 1024;
  if(MaxDataSize <= Kernel[12].BlockDim.x * Kernel[12].GridDim.x){
      MaxDataSize = Kernel[12].BlockDim.x * Kernel[12].GridDim.x;
  }

  for(int i = 0; i <= 12; i++){
    // fputs("Kernel ",fp);
    // sprintf(buffer, "%d", i);
    // fputs(i,fp);
    // fputs(": ",fp);
    sprintf(buffer, "%d", Kernel[i].BlockDim.x);
    fputs(buffer,fp);
    fputs("\t",fp);
    sprintf(buffer, "%d", Kernel[i].GridDim.x);
    fputs(buffer,fp);
    fputs("\n",fp);
  }

  fputs("\n",fp);

	// 
	// Kernel[13].reg = 0;
	// Kernel[13].sharedMem = 0;

  /*printf("Max data size: %d\n", MaxDataSize);
  for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
      printf("Kernel %d's thread: %d \t block: %d\n", i, Kernel[i].BlockDim.x, Kernel[i].GridDim.x);
    }*/
}

// Manual Set Kernel
extern void setKernel(Kernel_info *Kernel)
{
  Kernel[0].BlockDim =  dim3(512, 1, 1);
  Kernel[1].BlockDim =  dim3(256, 1, 1);
  Kernel[2].BlockDim =  dim3(576, 1, 1);
  Kernel[3].BlockDim =  dim3(1024, 1, 1);
  Kernel[4].BlockDim =  dim3(992, 1, 1);
  Kernel[5].BlockDim =  dim3(704, 1, 1);
  Kernel[6].BlockDim =  dim3(960, 1, 1); //running time varies
  Kernel[7].BlockDim =  dim3(352, 1, 1);
  Kernel[8].BlockDim =  dim3(928, 1, 1);
  Kernel[9].BlockDim =  dim3(288, 1, 1);
  Kernel[10].BlockDim = dim3(544, 1, 1);
  Kernel[11].BlockDim = dim3(832, 1, 1);
  Kernel[12].BlockDim = dim3(608, 1, 1);

  // int tempThread;
  for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
    // printf("Kernel %d' Thread:", i);
    // scanf("%d", &tempThread);
    // Kernel[i].BlockDim = dim3(tempThread, 1, 1);
    Kernel[i].GridDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].BlockDim.x), 1, 1);
    // printf("Kernel %d: %d %d\n", i, Kernel[i].BlockDim.x, Kernel[i].GridDim.x);
  }
}

extern void AlgoptKernel(Kernel_info *Kernel){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int N = TOTAL_KERNEL_NUM;
  int B = 16;
  int W = 32;

  int V[N][B][W];

  for(int b = 0; b<B; b++){
    for(int w = 0; w<W; w++){
      V[0][b][w] = 0;
    }
  }

  for(int n = 0; n < N; n++){
    V[n][0][0] = 65536;
  }

  for(int n = 0; n < N; n++){
    for(int b = 0; b<B; b++){
      for(int w = 0; w<W; w++){
        if(V[n][b-1][w] <= V[n-1][B-b][W-w]+Kernel[n].sharedMem*b){
          V[n][b][w] = V[n][b-1][w];
        }else{
          V[n][b][w] = V[n-1][B-b][W-w]+Kernel[n].sharedMem*b;
          // Kernel[n].GridDim = dim3(b,1,1);
          // Kernel[n].BlockDim = dim3(intCeil(Kernel[n].dataSize, Kernel[n].GridDim.x),1,1);
        }
      }
      if(V[n][b][W]<V[n][b-1][W]){
        B = B-b;
        W = W - intCeil(Kernel[n].dataSize, b*32);
      }
    }
  }
}

// Optimize Kernel
extern void optKernel(Kernel_info *Kernel)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
      if( intCeil(Kernel[i].dataSize, prop.multiProcessorCount) <= prop.maxThreadsPerBlock){
        Kernel[i].GridDim = dim3(prop.multiProcessorCount, 1, 1);
        Kernel[i].BlockDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].GridDim.x), 1, 1);
      }else{
        Kernel[i].BlockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
        Kernel[i].GridDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].BlockDim.x), 1, 1);
      }

      // if( Kernel[i].dataSize <= prop.maxThreadsPerBlock){
      //   Kernel[i].BlockDim = dim3(Kernel[i].dataSize, 1, 1);
      //   Kernel[i].GridDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].BlockDim.x), 1, 1);
      // }else{
      //   Kernel[i].BlockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
      //   Kernel[i].GridDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].BlockDim.x), 1, 1);
      // }

      Kernel[i].dataSize = Kernel[i].BlockDim.x * Kernel[i].GridDim.x;
      if(MaxDataSize <= Kernel[i].dataSize){
        MaxDataSize = Kernel[i].dataSize;
      }

      /*Kernel[1].sharedMem = sizeof(float) * 2 * Kernel[1].BlockDim.x;
      Kernel[3].sharedMem = sizeof(float) * 4 * Kernel[3].BlockDim.x;
      Kernel[7].sharedMem = (Kernel[7].BlockDim.x <= 32) ? 
          2 * Kernel[7].BlockDim.x * sizeof(float) : Kernel[7].BlockDim.x * sizeof(float);
      Kernel[8].sharedMem = (Kernel[8].BlockDim.x <= 32) ? 
          2 * Kernel[8].BlockDim.x * sizeof(float) : Kernel[8].BlockDim.x * sizeof(float);
      Kernel[9].sharedMem = (Kernel[9].BlockDim.x <= 32) ? 
          2 * Kernel[9].BlockDim.x * sizeof(float) : Kernel[9].BlockDim.x * sizeof(float);
      Kernel[10].sharedMem = (Kernel[10].BlockDim.x <= 32) ? 
          2 * Kernel[10].BlockDim.x * sizeof(float) : Kernel[10].BlockDim.x * sizeof(float);
*/
      
      //printf("%d's thread: %d \t block: %d\n", i, Kernel[i].BlockDim.x, Kernel[i].GridDim.x);
    }
}

// Using setting parameters for Kernel
extern void manualSetKernel(Kernel_info *Kernel, int Thread)
{

    for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
      Kernel[i].BlockDim = dim3(Thread, 1, 1);
      Kernel[i].GridDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].BlockDim.x), 1, 1);

      Kernel[i].dataSize = Kernel[i].BlockDim.x * Kernel[i].GridDim.x;
      if(MaxDataSize <= Kernel[i].dataSize){
        MaxDataSize = Kernel[i].dataSize;
      }
    }
}

// Choose a manual kernel to launch
extern void lauchManualKernel(int kernelNum, Kernel_info Kernel, float* D_data, float* D_result, 
  cudaStream_t* Streams, int StreamID)
{
  switch(kernelNum)
  {
    case 0:
    {
      KernelA<<<Kernel.GridDim, Kernel.BlockDim, Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result);
      break;
    }
    case 1:
    {
      KernelB<<<Kernel.GridDim, Kernel.BlockDim, Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result);
      break;
    }
    case 2:
    {
      KernelC<<<Kernel.GridDim, Kernel.BlockDim, Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result);
      break;
    }
    case 3:
    {
      KernelD<<<Kernel.GridDim, Kernel.BlockDim, Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result);
      break;
    }
    case 4:
    {
      KernelE<<<Kernel.GridDim, Kernel.BlockDim, Kernel.sharedMem, Streams[StreamID]>>>(D_data, D_result);
      break;
    }
  }
}

// Initialize Manual Kernels
extern void initManualKernel(Kernel_info *Kernel, int kernelMode, int Thread, int DataSize){
  time_t t;
  srand((unsigned) time(&t));

  // kernel A
  if(kernelMode == DEFAULT)
  {
    Kernel[0].dataSize = DATA_SIZE;
    Kernel[0].BlockDim = dim3(Thread,1,1);
  }else if(kernelMode == MANUAL){
    // int thread = 0;
    printf("Please input block size of kernel A: ");
    scanf("%d", &Thread);
    Kernel[0].BlockDim = dim3(Thread,1,1);

    printf("Please input total size of data: ");
    scanf("%d", &Kernel[0].dataSize);
    
  }else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[0].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[0].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[0].BlockDim = dim3(Thread,1,1);
  }

  Kernel[0].reg = 8;
  Kernel[0].GridDim = dim3(intCeil(Kernel[0].dataSize, Kernel[0].BlockDim.x),1,1);
  Kernel[0].sharedMem =  24576; //sizeof(float) * 6 * Kernel[1].BlockDim.x;
  Kernel[0].dataSize = Kernel[0].BlockDim.x * Kernel[0].GridDim.x;
  if(MaxDataSize <= Kernel[0].dataSize){
      MaxDataSize = Kernel[0].dataSize;
  }
  
  // kernel B
  if(kernelMode == DEFAULT)
  {
    Kernel[1].dataSize = DATA_SIZE;
    Kernel[1].BlockDim = dim3(Thread,1,1);
  }else if(kernelMode == MANUAL){
    // int thread = 0;
    printf("Please input block size of kernel B: ");
    scanf("%d", &Thread);
    Kernel[1].BlockDim = dim3(Thread,1,1);

    printf("Please input total size of data: ");
    scanf("%d", &Kernel[1].dataSize);

  }else if(kernelMode == RANDOM){ 
    if(DataSize != 0){
      Kernel[1].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[1].dataSize);

    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[1].BlockDim = dim3(Thread,1,1);
  }

  Kernel[1].reg = 16;
  Kernel[1].GridDim = dim3(intCeil(Kernel[1].dataSize, Kernel[1].BlockDim.x),1,1);
  Kernel[1].sharedMem =  12288; //sizeof(float) * 3 * Kernel[2].BlockDim.x;
  Kernel[1].dataSize = Kernel[1].BlockDim.x * Kernel[1].GridDim.x;
  if(MaxDataSize <= Kernel[1].dataSize){
      MaxDataSize = Kernel[1].dataSize;
  }

  // kernel C
  if(kernelMode == DEFAULT)
  {
    Kernel[2].dataSize = DATA_SIZE;
    Kernel[2].BlockDim = dim3(Thread,1,1);
  }else if(kernelMode == MANUAL){
    // int thread = 0;
    printf("Please input block size of kernel C: ");
    scanf("%d", &Thread);
    Kernel[2].BlockDim = dim3(Thread,1,1);

    printf("Please input total size of data: ");
    scanf("%d", &Kernel[2].dataSize);

  }else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[2].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[2].dataSize);

    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[2].BlockDim = dim3(Thread,1,1);
  }

  Kernel[2].reg = 24;
  Kernel[2].GridDim = dim3(intCeil(Kernel[2].dataSize, Kernel[2].BlockDim.x),1,1);
  Kernel[2].sharedMem =  6144; //sizeof(float) * 2 * Kernel[2].BlockDim.x;
  Kernel[2].dataSize = Kernel[2].BlockDim.x * Kernel[2].GridDim.x;
  if(MaxDataSize <= Kernel[2].dataSize){
      MaxDataSize = Kernel[2].dataSize;
  }

  // kernel D
  if(kernelMode == DEFAULT)
  {
    Kernel[3].dataSize = DATA_SIZE;
    Kernel[3].BlockDim = dim3(Thread,1,1);
  }else if(kernelMode == MANUAL){
    // int thread = 0;
    printf("Please input block size of kernel D: ");
    scanf("%d", &Thread);
    Kernel[3].BlockDim = dim3(Thread,1,1);

    printf("Please input total size of data: ");
    scanf("%d", &Kernel[3].dataSize);

  }else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[3].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[3].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[3].BlockDim = dim3(Thread,1,1);
  }

  Kernel[3].reg = 32;
  Kernel[3].GridDim = dim3(intCeil(Kernel[3].dataSize, Kernel[3].BlockDim.x),1,1);
  Kernel[3].sharedMem =  4608; //sizeof(float) * 1 * Kernel[2].BlockDim.x;
  Kernel[3].dataSize = Kernel[3].BlockDim.x * Kernel[3].GridDim.x;
  if(MaxDataSize <= Kernel[3].dataSize){
      MaxDataSize = Kernel[3].dataSize;
  }

  // kernel E
  if(kernelMode == DEFAULT)
  {
    Kernel[4].dataSize = DATA_SIZE;
    Kernel[4].BlockDim = dim3(Thread,1,1);
  }else if(kernelMode == MANUAL){
    // int thread = 0;
    printf("Please input block size of kernel 5 (increment_kernel): ");
    scanf("%d", &Thread);
    Kernel[4].BlockDim = dim3(Thread,1,1);

    printf("Please input total size of data: ");
    scanf("%d", &Kernel[4].dataSize);

  }else if(kernelMode == RANDOM){
    if(DataSize != 0){
      Kernel[4].dataSize = DataSize;
    }else{
      printf("Please input total size of data: ");
      scanf("%d", &Kernel[4].dataSize);
    }

    Thread = (rand()%(32) + 1) * 32;
    Kernel[4].BlockDim = dim3(Thread,1,1);
  }

  Kernel[4].reg = 40;
  Kernel[4].GridDim = dim3(intCeil(Kernel[4].dataSize, Kernel[4].BlockDim.x),1,1);
  Kernel[4].sharedMem =  1536; //sizeof(float) * 0 * Kernel[2].BlockDim.x;
  Kernel[4].dataSize = Kernel[4].BlockDim.x * Kernel[4].GridDim.x;
  if(MaxDataSize <= Kernel[4].dataSize){
      MaxDataSize = Kernel[4].dataSize;
  } 
}

// Optimize Manual Kernel
extern void optManualKernel(Kernel_info *Kernel)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
      if( intCeil(Kernel[i].dataSize, prop.multiProcessorCount) <= prop.maxThreadsPerBlock){
        Kernel[i].GridDim = dim3(prop.multiProcessorCount, 1, 1);
        Kernel[i].BlockDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].GridDim.x), 1, 1);
      }else{
        Kernel[i].BlockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
        Kernel[i].GridDim = dim3(intCeil(Kernel[i].dataSize, Kernel[i].BlockDim.x), 1, 1);
      }

      Kernel[i].dataSize = Kernel[i].BlockDim.x * Kernel[i].GridDim.x;
      if(MaxDataSize <= Kernel[i].dataSize){
        MaxDataSize = Kernel[i].dataSize;
      }

      Kernel[0].sharedMem =  16504; //sizeof(float) * 6 * Kernel[1].BlockDim.x;
      Kernel[1].sharedMem =  8252; //sizeof(float) * 3 * Kernel[2].BlockDim.x;
      Kernel[2].sharedMem =  4126; //sizeof(float) * 2 * Kernel[2].BlockDim.x;
      Kernel[3].sharedMem =  2063; //sizeof(float) * 1 * Kernel[2].BlockDim.x;
      Kernel[4].sharedMem =  0; //sizeof(float) * 0 * Kernel[2].BlockDim.x;
      
      // printf("%d's thread: %d \t block: %d\n", i, Kernel[i].BlockDim.x, Kernel[i].GridDim.x);
    }
}

// Run kernels squentially
extern float sequenceRun(int TotalKernelNum, int *KernelNum, Kernel_info *Kernel, 
  float *D_data, float *D_result, cudaStream_t *Streams){

  float msecTotal = 0.0f;

  cudaEvent_t start;
  error = cudaEventCreate(&start);
  
  if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    }

  cudaEvent_t start_single_kernel;
  cudaEventCreate(&start_single_kernel);
  cudaEvent_t stop_single_kernel;
  cudaEventCreate(&stop_single_kernel);

  // Record the start event
    error = cudaEventRecord(start, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

  // Invoke kernels one by one
  for(int i = 0; i <TotalKernelNum; i++){
    // msecTotal += lauchKernel(KernelNum[i], Kernel[i], D_data, D_result, Streams, 0);
    
    /*
     *  void lauchKernel(int kernelNum, Kernel_info Kernel, float* D_data, float* D_result, 
     *                    cudaStream_t* Streams, int StreamID);
     */
    lauchKernel(KernelNum[i], Kernel[KernelNum[i]], D_data, D_result, Streams, 0);
  // cudaEventRecord(start_single_kernel, 0);

    // lauchManualKernel(KernelNum[i], Kernel[KernelNum[i]], D_data, D_result, Streams, 0);
    
    // cudaEventRecord(stop_single_kernel, 0);
    // cudaEventSynchronize(stop_single_kernel);

    // float msec = 0.0f;
    // cudaEventElapsedTime(&msec, start_single_kernel, stop_single_kernel);

    // printf("Kernel %d: %f\n", KernelNum[i], msec);

    //printf("launching Kernel:%d\n", KernelNum[i]);
    // int k = 0;
    // lauchKernel(KernelNum[k], Kernel[KernelNum[k]], D_data, D_result, Streams, 0);

  }

  // lauchKernel(5, Kernel[5], D_data, D_result, Streams, 0);
  // lauchKernel(11, Kernel[11], D_data, D_result, Streams, 0);  

  // Record the stop event
    error = cudaEventRecord(stop, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    // printf("Sequential Running Time: %f ms\n", msecTotal);

  return msecTotal;
}

// Run kernels concurrently
extern float concurrentRun(int TotalKernelNum, int *KernelNum, Kernel_info *Kernel, 
	float *D_data, float *D_result, cudaStream_t *Streams){

	float msecTotal = 0.0f;

	cudaEvent_t start;
	error = cudaEventCreate(&start);
	
	if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    }

	// Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

	for(int i = 0; i <TotalKernelNum; i++){
		// msecTotal += lauchKernel(KernelNum[i], Kernel[i], D_data, D_result, Streams, i);

    /*
     *  void lauchKernel(int kernelNum, Kernel_info Kernel, float* D_data, float* D_result, 
     *                    cudaStream_t* Streams, int StreamID);
     */
    // printf("lauch %dth Kernel\n", i);
		lauchKernel(KernelNum[i], Kernel[KernelNum[i]], D_data, D_result, Streams, i%32);
    // lauchManualKernel(KernelNum[i], Kernel[KernelNum[i]], D_data, D_result, Streams, i);
	}

	// Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

	// printf("Concurrent Running Time: %f ms\n", msecTotal);

	return msecTotal;
}

// Run with default parameters
extern int defaultRun(int Thread, int DataSize){

  if(Thread == 0){
	 Thread = DEFAULT_THREADS;
  }

  if(DataSize == 0) {
	 DataSize = DATA_SIZE;
  }

  // MaxDataSize = intCeil(DataSize, 1024) * 1024;

	// Kernel infomantion
	Kernel_info kernel[TOTAL_KERNEL_NUM];
	initKernel(kernel, DEFAULT, Thread, DataSize);

	// Choose which kernels to run
	int *kernelNum;
	kernelNum = (int *)malloc(sizeof(int) * TOTAL_KERNEL_NUM);
	for(int i = 0; i < 31; i++)
	{
		kernelNum[i] = i % TOTAL_KERNEL_NUM;
	}

	// Host data and result
	float *h_data, *h_result;

  // Device data and result
  float *d_data, *d_result;

	// Allocate space for host data and result
	h_data = (float *)malloc(sizeof(float) * MaxDataSize * 1024);
	h_result = (float *)malloc(sizeof(float) * MaxDataSize * 1024);

	// Initilized host data 
	constantInit(h_data, MaxDataSize * 1024);
	constantInit(h_result, MaxDataSize * 1024);

	error = cudaMalloc((void **) &d_data, sizeof(float) * MaxDataSize);

	if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_result, sizeof(float) * MaxDataSize);

	if (error != cudaSuccess)
  {
   	printf("cudaMalloc (d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
   	exit(EXIT_FAILURE);
 	}

 	// Copy data from host to device
 	error = cudaMemcpy(d_data, h_data, sizeof(float) * MaxDataSize, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_data,h_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // allocate and initialize an array of stream handles
  //cudaStream_t *streams = (cudaStream_t *) malloc(TOTAL_KERNEL_NUM * sizeof(cudaStream_t));
 cudaStream_t *streams = (cudaStream_t *) malloc(31 * sizeof(cudaStream_t));

  //for (int i = 0; i < (TOTAL_KERNEL_NUM + 1); i++)
  for (int i = 0; i < (32); i++)
  {
    error = cudaStreamCreate(&(streams[i]));

    if (error != cudaSuccess)
    {
      printf("cudaStreamCreate (streams[i]) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  // Run kernels sequentially
  // sequenceRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams);
  //printf("Sequential Running Time: %f ms\n", sequenceRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams));
	printf("Sequential Running Time: %f ms\n", sequenceRun(31, kernelNum, kernel, d_data, d_result, streams));

  // Run kernels concurrently
  // concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams);
  //printf("Concurrent Running Time: %f ms\n", concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams));
	printf("Concurrent Running Time: %f ms\n", concurrentRun(31, kernelNum, kernel, d_data, d_result, streams));

printf("1\n");

   // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[TOTAL_KERNEL_NUM]);

printf("2\n");

  if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

printf("3\n");

  //error = cudaFree(d_data);

printf("4\n");

 // error = cudaFree(d_result);


printf("5\n");

  // Run kernels optimized
  optKernel(kernel);

printf("6\n");

  error = cudaMalloc((void **) &d_data, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_result, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // Copy data from host to device
  error = cudaMemcpy(d_data, h_data, sizeof(float) * MaxDataSize, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_data,h_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  //printf("Optimized Running Time: %f ms\n", concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams));
	printf("Optimized Running Time: %f ms\n", concurrentRun(31, kernelNum, kernel, d_data, d_result, streams));

  // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[TOTAL_KERNEL_NUM]);

	if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < TOTAL_KERNEL_NUM; i++)
  {
    error = cudaStreamDestroy(streams[i]);
  }

  // Clean up memory
  free(streams);

  free(h_data);
	free(h_result);

	error = cudaFree(d_data);

	if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_data (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

	error = cudaFree(d_result);

	if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_result (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

	return 0;
}

// Running random number of kernels and input each kernel's informantion manually
extern int manualRun(int Num){

	int thread = DEFAULT_THREADS;
	int dataSize = DATA_SIZE;
	MaxDataSize = DATA_SIZE;

	// Kernel infomantion
	Kernel_info kernel[TOTAL_KERNEL_NUM];
	initKernel(kernel, DEFAULT, thread, dataSize);

	// Choose which kernels to run
	int *kernelNum;
	kernelNum = (int *)malloc(sizeof(int) * Num);
	
	time_t t;
	srand((unsigned) time(&t));
	for(int i = 0; i < Num; i++)
	{
		kernelNum[i] = rand()%(13);
		printf("%d\n", kernelNum[i]);
	}

	// Host data and result
  float *h_data, *h_result;

  // Device data and result
  float *d_data, *d_result;

  // Allocate space for host data and result
  h_data = (float *)malloc(sizeof(float) * MaxDataSize * 1024);
  h_result = (float *)malloc(sizeof(float) * MaxDataSize * 1024);

  // Initilized host data 
  constantInit(h_data, MaxDataSize * 1024);

  error = cudaMalloc((void **) &d_data, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_result, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // Copy data from host to device
  error = cudaMemcpy(d_data, h_data, sizeof(float) * MaxDataSize, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_data,h_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // allocate and initialize an array of stream handles
  cudaStream_t *streams = (cudaStream_t *) malloc(TOTAL_KERNEL_NUM * sizeof(cudaStream_t));

  for (int i = 0; i < (TOTAL_KERNEL_NUM + 1); i++)
  {
    error = cudaStreamCreate(&(streams[i]));

    if (error != cudaSuccess)
    {
      printf("cudaStreamCreate (streams[i]) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  // Run kernels sequentially
  // sequenceRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams);
  printf("Sequential Running Time: %f ms\n", sequenceRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams));

  // Run kernels concurrently
  // concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams);
  printf("Concurrent Running Time: %f ms\n", concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams));

   // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[TOTAL_KERNEL_NUM]);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_data);

  if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_data (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_result);

  if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_result (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // Run kernels optimized
  optKernel(kernel);

  error = cudaMalloc((void **) &d_data, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_result, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // Copy data from host to device
  error = cudaMemcpy(d_data, h_data, sizeof(float) * MaxDataSize, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_data,h_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  printf("Optimized Running Time: %f ms\n", concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams));

  // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[TOTAL_KERNEL_NUM]);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < TOTAL_KERNEL_NUM; i++)
  {
    error = cudaStreamDestroy(streams[i]);
  }

  // Clean up memory
  free(streams);

  free(h_data);
  free(h_result);

  error = cudaFree(d_data);

  if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_data (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_result);

  if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_result (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  return 0;
}

// Running random number of kernels with random block number and block size
extern int benchmarkRun(int Num, int Thread, int Datasize){

  // int thread = Thread;
  // int dataSize = Datasize;
  MaxDataSize = 5000;//Datasize;

  // Kernel infomantion
  Kernel_info kernel[TOTAL_KERNEL_NUM];
  initKernel(kernel, RANDOM, Thread, Datasize);
  // initManualKernel(kernel, RANDOM, Thread, Datasize);

  // Choose which kernels to run
  int *kernelNum;
  kernelNum = (int *)malloc(sizeof(int) * Num);

  // int tempI = 0;
  // for(int i = 0; i < Num; i++)
  // {
  //   if(tempI%13 == 6){
  //     tempI++;
  //   }    
  //   kernelNum[i] = tempI%13;
  //   tempI++;
  //   // printf("%d\n", kernelNum[i]);
  // }

  // if(Num == 16){
  //   kernelNum[0] = 4;
  //   kernelNum[1] = 6;
  //   kernelNum[2] = 0;
  //   kernelNum[3] = 7;
  //   kernelNum[4] = 2;
  //   kernelNum[5] = 4;
  //   kernelNum[6] = 10;
  //   kernelNum[7] = 4;
  //   kernelNum[8] = 1;
  //   kernelNum[9] = 8;
  //   kernelNum[10] = 3;
  //   kernelNum[11] = 3;
  //   kernelNum[12] = 2;
  //   kernelNum[13] = 3;
  //   kernelNum[14] = 5;
  //   kernelNum[15] = 5;
  //   // kernelNum[16] = 12;
  //   // kernelNum[17] = 9;
  //   // kernelNum[18] = 1;
  //   // kernelNum[19] = 9;
  //   // kernelNum[20] = 12;
  //   // kernelNum[21] = 7;
  //   // kernelNum[22] = 11;
  //   // kernelNum[23] = 11;
  //   // kernelNum[24] = 11;
  //   // kernelNum[25] = 3;
  //   // kernelNum[26] = 3;
  //   // kernelNum[27] = 8;
  //   // kernelNum[28] = 10;
  //   // kernelNum[29] = 3;
  //   // kernelNum[30] = 12;
  // }

  for(int i = 0; i < Num; i++)
  {
    kernelNum[i] = i % TOTAL_KERNEL_NUM;  
    // printf("%d\n", kernelNum[i]);   
  //   kernelNum[i] = rand()%(13);
  // while(kernel[kernelNum[i]].sharedMem == 0){
  //  kernelNum[i] = rand()%(13); 
  }
  
 //  time_t t;
 //  srand((unsigned) time(&t));
 //  printf("\n");
 //  for(int i = 0; i < Num; i++)
 //  {
 //    //kernelNum[i] = i % TOTAL_KERNEL_NUM;     
	//   kernelNum[i] = rand()%(13);
	// while(kernel[kernelNum[i]].sharedMem == 0){
	// 	kernelNum[i] = rand()%(13);	
	// }
 //    printf("%d ", kernelNum[i]);
 //  }
 //  printf(":\n");

  // Run kernels with FIXED size of thread
  //manualSetKernel(kernel, (rand()%(32) + 1) * 32);

  // Host data and result
  float *h_data, *h_result;

  // Device data and result
  float *d_data, *d_result;

  // clock_t begin = clock();
// AlgoptKernel(kernel);
// clock_t end = clock();
// double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
// printf("time:%f ms\n", time_spent*1000);

  // Allocate space for host data and result
  h_data = (float *)malloc(sizeof(float) * MaxDataSize * 1024);
  h_result = (float *)malloc(sizeof(float) * MaxDataSize * 1024);

  // Initilized host data 
  constantInit(h_data, MaxDataSize * 1024);

  error = cudaMalloc((void **) &d_data, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_result, sizeof(float) * MaxDataSize);

  if (error != cudaSuccess)
  {
    printf("cudaMalloc (d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // Copy data from host to device
  error = cudaMemcpy(d_data, h_data, sizeof(float) * MaxDataSize, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_data,h_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  int maxStreamNum = 32;
  if(Num <= 32){
    maxStreamNum = Num;
  }

  // allocate and initialize an array of stream handles
  cudaStream_t *streams = (cudaStream_t *) malloc(maxStreamNum * sizeof(cudaStream_t));
  for (int i = 0; i < maxStreamNum; i++)
  {
    error = cudaStreamCreate(&(streams[i]));

    if (error != cudaSuccess)
    {
      printf("cudaStreamCreate (streams[i]) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  // for(int it = 0; it < 400; it ++)
  {

    // initKernel(kernel, RANDOM, Thread, Datasize);

  // Run kernels sequentially
  // sequenceRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams);

  // printf("sequenceRun start\n");
  printf("%f\t", sequenceRun(Num, kernelNum, kernel, d_data, d_result, streams));
  // printf("sequenceRun end\n");

   // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[0]);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  int countKernel[13];
  for(int i = 0; i < 13; i++){
    countKernel[i] = 0;
  }
  int count = 0;
  // time_t t;
  // srand((unsigned) time(&t));
  while(count < Num){
    // printf("%d: ", count);
    do{
      kernelNum[count] = rand()%(13);
      // printf("rand:%d\t",kernelNum[count]);
      countKernel[kernelNum[count]] ++; 
    }while(countKernel[kernelNum[count]] > ceil(Num/13));
    // printf("adopt kernel:%d\n",kernelNum[count]);
    // printf("%d\n",kernelNum[count]);
    count++;
  }

  // setKernel(kernel);
  // optKernel(kernel);
  // Run kernels concurrently
  // concurrentRun(TOTAL_KERNEL_NUM, kernelNum, kernel, d_data, d_result, streams);
  // printf("random concurrentRun start\n");
  printf("%f\t", concurrentRun(Num, kernelNum, kernel, d_data, d_result, streams));
  // printf("random concurrentRun end\n");

   // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[0]);

  if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  // error = cudaFree(d_data);

  // if (error != cudaSuccess)
  // {
  //   fprintf(stderr, "Failed to free device d_data (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
  //   exit(EXIT_FAILURE);
  // }

  // error = cudaFree(d_result);

  // if (error != cudaSuccess)
  // {
  //   fprintf(stderr, "Failed to free device d_result (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
  //   exit(EXIT_FAILURE);
  // }

  // Run kernels optimized
  optKernel(kernel);
  // optManualKernel(kernel);

  // error = cudaMalloc((void **) &d_data, sizeof(float) * MaxDataSize);

  // if (error != cudaSuccess)
  // {
  //   printf("cudaMalloc (d_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
  //   exit(EXIT_FAILURE);
  // }

  // error = cudaMalloc((void **) &d_result, sizeof(float) * MaxDataSize);

  // if (error != cudaSuccess)
  // {
  //   printf("cudaMalloc (d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
  //   exit(EXIT_FAILURE);
  // }

  // // Copy data from host to device
  // error = cudaMemcpy(d_data, h_data, sizeof(float) * MaxDataSize, cudaMemcpyHostToDevice);

  // if (error != cudaSuccess)
  // {
  //   printf("cudaMemcpy (d_data,h_data) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
  //   exit(EXIT_FAILURE);
  // }

  printf(" %f\t", concurrentRun(Num, kernelNum, kernel, d_data, d_result, streams));

  // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[0]);


if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  if(Num == 39){
    int k = 0;
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 3; k++;
    }
    kernelNum[k] = 7; k++;
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 0; k++;
      kernelNum[k] = 4; k++;
      kernelNum[k] = 5; k++;
    }

    for(int i = 0; i < 2; i++){
      kernelNum[k] = 7; k++;
    }

    for(int i = 0; i < 3; i++){
      kernelNum[k] = 2; k++;
      kernelNum[k] = 6; k++;
    }

    for(int i = 0; i < 3; i++){
      kernelNum[k] = 10;k++; 
      kernelNum[k] = 11;k++;
      kernelNum[k] = 12;k++;
    }

    for(int i = 0; i < 3; i++){
      kernelNum[k] = 1;k++;
      kernelNum[k] = 8;k++;
      kernelNum[k] = 9;k++;
    }

    // printf("%d\n", k);

  }else if(Num == 52){
    int k = 0;
    for(int i = 0; i < 4; i++){
      kernelNum[k] = 2;k++;
      kernelNum[k] = 6;k++;
    }
    for(int i = 0; i < 4; i++){
      kernelNum[k] = 3;k++;
    }
    kernelNum[k] = 0;k++;
    kernelNum[k] = 4;k++;
    kernelNum[k] = 5;k++;

    for(int i = 0; i < 4; i++){
      kernelNum[k] = 7;k++;
    }
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 0;k++;
      kernelNum[k] = 4;k++;
      kernelNum[k] = 5;k++;
    }
   

    for(int i = 0; i < 4; i++){
      kernelNum[k] = 10;k++;
      kernelNum[k] = 11;k++;
      kernelNum[k] = 12;k++;
    }

    for(int i = 0; i < 4; i++){
      kernelNum[k] = 1;k++;
      kernelNum[k] = 8;k++;
      kernelNum[k] = 9;k++;
    }
    // printf("%d\n", k);

  }else if(Num == 65){
    int k = 0;

    for(int i = 0; i < 5; i++){
      kernelNum[k] = 6;k++;
    }
    for(int i = 0; i < 5; i++){
      kernelNum[k] = 2;k++;
    }

    for(int i = 0; i < 4; i++){
      kernelNum[k] = 3;k++;
    }
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 0;k++;
      kernelNum[k] = 4;k++;
      kernelNum[k] = 5;k++;
    }

    kernelNum[k] = 3;k++;
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 7;k++;
    }
    for(int i = 0; i < 2; i++){
      kernelNum[k] = 0;k++;
      kernelNum[k] = 4;k++;
      kernelNum[k] = 5;k++;
    }

    for(int i = 0; i < 2; i++){
      kernelNum[k] = 7;k++;
    }

    for(int i = 0; i < 5; i++){
      kernelNum[k] = 10;k++;
      kernelNum[k] = 11;k++;
      kernelNum[k] = 12;k++;
    }

    for(int i = 0; i < 5; i++){
      kernelNum[k] = 1;k++;
      kernelNum[k] = 8;k++;
      kernelNum[k] = 9;k++;
    }
    // printf("%d\n", k);

  }else if(Num == 78){
    int k = 0;

    for(int i = 0; i < 4; i++){
      kernelNum[k] = 3;k++;
    }
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 0;k++;
      kernelNum[k] = 4;k++;
      kernelNum[k] = 5;k++;
    }

    for(int i = 0; i < 2; i++){
      kernelNum[k] = 3;k++;
    }
    for(int i = 0; i < 2; i++){
      kernelNum[k] = 7;k++;
    }
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 0;k++;
      kernelNum[k] = 4;k++;
      kernelNum[k] = 5;k++;
    }

    for(int i = 0; i < 4; i++){
      kernelNum[k] = 7;k++;
    }
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 2;k++;
      kernelNum[k] = 6;k++;
    }

    for(int i = 0; i < 6; i++){
      kernelNum[k] = 10;k++;
      kernelNum[k] = 11;k++;
      kernelNum[k] = 12;k++;
    }
    for(int i = 0; i < 3; i++){
      kernelNum[k] = 2;k++;
      kernelNum[k] = 6;k++;
    }
    for(int i = 0; i < 6; i++){
      kernelNum[k] = 1;k++;
      kernelNum[k] = 8;k++;
      kernelNum[k] = 9;k++;
    }
    // printf("%d\n", k);
  }
  
  // printf("binpack concurrentRun start\n");
  printf(" %f\n", concurrentRun(Num, kernelNum, kernel, d_data, d_result, streams));
  // printf("binpack concurrentRun end\n");

  // Copy result back from device to host
  error = cudaMemcpyAsync(h_result, d_result, sizeof(float) * MaxDataSize, cudaMemcpyDeviceToHost, streams[0]);

if (error != cudaSuccess)
  {
    printf("cudaMemcpyAsync (h_result,d_result) returned error code (%s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  }

  free(h_data);
  free(h_result);

  error = cudaFree(d_data);

  if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_data (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_result);

  if (error != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device d_result (error code %s), line(%d)\n", cudaGetErrorString(error), __LINE__);
    exit(EXIT_FAILURE);
  }

for (int i = 0; i < maxStreamNum; i++)
  {
    error = cudaStreamDestroy(streams[i]);
  }

  // Clean up memory
  free(streams);

  return 0;
}


// Based on command line argument to choose different functions to run
extern int primitiveFunc(int argc, char **argv)
{
	// if(argc > 3){
	// 	printf("Command Line Argument:\n\n");
	// 	printf("--default\t\t\t\t(-d) \n\t running all the kernels with default block size and block number.\n");
	// 	printf("--default [block size]\t\t\t\t(-d [block size]) \n\t running all the kernels with input block size .\n");
	// 	printf("--default [block size] [data size]\t\t\t\t(-d [block size] [data size]) \n\t running all the kernels with input block size and data size.\n");
	// 	printf("--help\t\t\t\t(-h) \n\t Show user help.\n\n");
	// 	printf("--manual [num]\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
	// 	printf("--benchmark [num]\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n");
	// 	return 1;
	// }else 

  /*
   *  Running all the kernels with default setttings.
   */
	if((strcmp(argv[1], "--default")==0) || (strcmp(argv[1], "-d")==0))
	{
		if(argc == 2){
			int errorCode = defaultRun(DEFAULT_THREADS, DATA_SIZE);

			if(errorCode != 0){
				printf("defaultRun return error code %d, line (%d)\n", errorCode, __LINE__);
			}

			return 0;
		}
		else if(argc == 3){
			int errorCode = defaultRun(atoi(argv[2]), DATA_SIZE);

			if(errorCode != 0){
				printf("defaultRun return error code %d, line (%d)\n", errorCode, __LINE__);
			}

			return 0;
		}
		else if(argc == 4){
			int errorCode = defaultRun(atoi(argv[2]), atoi(argv[3]));

			if(errorCode != 0){
				printf("defaultRun return error code %d, line (%d)\n", errorCode, __LINE__);
			}

			return 0;
		}
		else
		{
			printf("Command Line Argument:\n\n");
			printf("--default\t\t\t\t(-d) \n\t running all the kernels with default block size and block number.\n\n");
			printf("--default [block size]\t\t\t(-d [block size]) \n\t running all the kernels with input block size .\n\n");
			printf("--default [block size] [data size]\t(-d [block size] [data size]) \n\t running all the kernels with input block size and data size.\n\n");
			printf("--help\t\t\t\t\t(-h) \n\t Show user help.\n\n");
			printf("--manual [num]\t\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
			printf("--benchmark [num]\t\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n\n");
			printf("--benchmark [num] [data size]\t\t(-b [num] [data size]) \n\t Running random number of kernels with same input data size, got the results of both assigning kernel info ramdom and optimized.\n\n");
      return 1;
		}
	}
  /*
   *  Running kernels with manual setting.
   */
  else if((strcmp(argv[1], "--manual")==0) || (strcmp(argv[1], "-m")==0))
  {
    if(argc == 3)
    {
      if(atoi(argv[2]) < 0 || atoi(argv[2]) > TOTAL_KERNEL_NUM){
        printf("The total number of kernels should be 1~%d\n", TOTAL_KERNEL_NUM);
        
        return 1;
      }else{
        manualRun(atoi(argv[2]));

        return 0;
      }
    }else{
      printf("Command Line Argument:\n\n");
      printf("--default\t\t\t\t(-d) \n\t running all the kernels with default block size and block number.\n\n");
      printf("--default [block size]\t\t\t(-d [block size]) \n\t running all the kernels with input block size .\n\n");
      printf("--default [block size] [data size]\t(-d [block size] [data size]) \n\t running all the kernels with input block size and data size.\n\n");
      printf("--help\t\t\t\t\t(-h) \n\t Show user help.\n\n");
      printf("--manual [num]\t\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
      printf("--benchmark [num]\t\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n\n");
      printf("--benchmark [num] [data size]\t\t(-b [num] [data size]) \n\t Running random number of kernels with same input data size, got the results of both assigning kernel info ramdom and optimized.\n\n");
      return 1;  
    }
  }
  /*
   *  Running kernels with benchmark.
   */
  else if((strcmp(argv[1], "--benchmark")==0) || (strcmp(argv[1], "-b")==0))
  {
    if(argc == 3)
    {
      int errorCode = benchmarkRun(atoi(argv[2]), 0, 0);

      if(errorCode != 0){
        printf("defaultRun return error code %d, line (%d)\n", errorCode, __LINE__);
      }

      return 0;
    }
    else if(argc == 4){
      int errorCode = benchmarkRun(atoi(argv[2]), 0, atoi(argv[3]));

      if(errorCode != 0){
        printf("defaultRun return error code %d, line (%d)\n", errorCode, __LINE__);
      }

      return 0;
    }else{
      printf("Command Line Argument:\n\n");
      printf("--default\t\t\t\t(-d) \n\t running all the kernels with default block size and block number.\n\n");
      printf("--default [block size]\t\t\t(-d [block size]) \n\t running all the kernels with input block size .\n\n");
      printf("--default [block size] [data size]\t(-d [block size] [data size]) \n\t running all the kernels with input block size and data size.\n\n");
      printf("--help\t\t\t\t\t(-h) \n\t Show user help.\n\n");
      printf("--manual [num]\t\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
      printf("--benchmark [num]\t\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n\n");
      printf("--benchmark [num] [data size]\t\t(-b [num] [data size]) \n\t Running random number of kernels with same input data size, got the results of both assigning kernel info ramdom and optimized.\n\n");
      return 1;  
    }
  }
  /*
   *  Show help
   */
	else if((strcmp(argv[1], "--help")==0) || (strcmp(argv[1], "-h")==0))
	{
		printf("Command Line Argument:\n\n");
		printf("--default\t\t\t\t(-d) \n\t running all the kernels with default block size and block number.\n\n");
		printf("--default [block size]\t\t\t(-d [block size]) \n\t running all the kernels with input block size .\n\n");
		printf("--default [block size] [data size]\t(-d [block size] [data size]) \n\t running all the kernels with input block size and data size.\n\n");
		printf("--help\t\t\t\t\t(-h) \n\t Show user help.\n\n");
		printf("--manual [num]\t\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
		printf("--benchmark [num]\t\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n");
		printf("--benchmark [num] [data size]\t\t(-b [num] [data size]) \n\t Running random number of kernels with same input data size, got the results of both assigning kernel info ramdom and optimized.\n\n");
    return 0;
	}
	
	else
	{
		printf("Command Line Argument:\n\n");
		printf("--default\t\t\t\t(-d) \n\t running all the kernels with default block size and block number.\n\n");
		printf("--default [block size]\t\t\t(-d [block size]) \n\t running all the kernels with input block size .\n\n");
		printf("--default [block size] [data size]\t(-d [block size] [data size]) \n\t running all the kernels with input block size and data size.\n\n");
		printf("--help\t\t\t\t\t(-h) \n\t Show user help.\n\n");
		printf("--manual [num]\t\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
		printf("--benchmark [num]\t\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n");
    printf("--benchmark [num] [data size]\t\t(-b [num] [data size]) \n\t Running random number of kernels with same input data size, got the results of both assigning kernel info ramdom and optimized.\n\n");
		return 1;
	}
}

#endif /* FUNCTIONS_H */
