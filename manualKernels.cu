/*-----------
 *
 * clock.cu
 *
 * This is the source file of a kernel to measure time for each block.
 *
 * These kernels are man made kernels to have specific attribute.
 *
 * streamsOptBenchmark/mannualKernels.cu
 *
 * By Hao Li
 *
 *------------
 */

#include <stdlib.h>
#include <stdio.h>
#include "functions.cuh"

__global__ void KernelA(const float *input, float *output) //reg 1-8
{
	for(int l = 0; l < 1000000; l++){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

    	output[id] = input[id];
    }
}

__global__ void KernelB(const float *input, float *output) // reg 9-16
{
	for(int l = 0; l < 100; l++){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

		int k = 0;

    	for (int i=0; i<16384; i++){
    		k += input[id];
    		k %= 1024;
    	}

    	output[id] = k;
    }
}

__global__ void KernelC(const float *input, float *output) // reg 17-24
{
	for(int l = 0; l < 100; l++){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

		int k = 0;

    	for (int i=0; i<16384; i++){
    		k += input[id];
    		k %= 1024;
    		k -= (input[id] / threadIdx.x);
    	}

    	output[id] = k;
    }
}

__global__ void KernelD(const float *input, float *output) // reg 25-32
{
	for(int l = 0; l < 20; l++){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

		int k = 0;

		int j[10];

    	for (int i=0; i<16384; i++){
    		k += input[id]*blockIdx.x;
    		k %= 1024;

    		for(int n = 0; n < 10; n++){
    			j[n] += input[id];
    			j[n] %= 1024;
    			k += j[n];
    		}
    	}

    	output[id] = k;
    }
}

__global__ void KernelE(const float *input, float *output) // reg 33-40
{
	for(int l = 0; l < 10; l++){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

		int k = 0;

		int j[10];
		int m[10];

    	for (int i=0; i<16384; i++){
    		k += input[id]*blockIdx.x;
    		k %= 1024;

    		for(int n = 0; n < 10; n++){
    			j[n] += input[id];
    			j[n] %= 1024;
    			k += j[n];
    		}
    	}

    	for (int i=0; i<16384; i++){
    		for(int n = 0; n < 10; n++){
    			m[n] += input[id];
    			m[n] %= 1024;
    			k += m[n];
    		}
    	}

    	output[id] = k;
    }
}

__global__ void KernelF(const float *input, float *output) // reg 41-48
{
	for(int l = 0; l < 100; l++){
		int id = blockDim.x * blockIdx.x + threadIdx.x;

		int k = 0;

		int j[10];
		int m[10];
		int o[10];

    	for (int i=0; i<16384; i++){
    		k += input[id]*blockIdx.x;
    		k %= 1024;

    		for(int n = 0; n < 10; n++){
    			j[n] += input[id];
    			j[n] %= 1024;
    			k += j[n];
    		}
    	}

    	for (int i=0; i<16384; i++){
    		for(int n = 0; n < 10; n++){
    			m[n] += input[id];
    			m[n] %= 1024;
    			k += m[n];
    		}
    	}

    	for (int i=0; i<16384; i++){
    		for(int n = 0; n < 10; n++){
    			o[n] *= input[id];
    			o[n] %= 1024;
    			k += o[n];
    		}
    	}

    	output[id] = k;
    }
}

__global__ void KernelG(const float *input, float *output) // reg 49-56
{

}

__global__ void KernelH(const float *input, float *output) // reg 57-64
{

}

// int main(int argc, char **argv){

// 	int dataSize = 100;

// 	// Host data and result
// 	float *h_data, *h_result;

//   	// Device data and result
//   	float *d_data, *d_result;

// 	// Allocate space for host data and result
// 	h_data = (float *)malloc(sizeof(float) * dataSize * 1024);
// 	h_result = (float *)malloc(sizeof(float) * dataSize * 1024);

// 	// Initilized host data 
// 	constantInit(h_data, dataSize * 1024);

// 	cudaMalloc((void **) &d_data, sizeof(float) * dataSize);
// 	cudaMalloc((void **) &d_result, sizeof(float) * dataSize);

// 	cudaMemcpy(d_data, h_data, sizeof(float) * dataSize, cudaMemcpyHostToDevice);

// 	// KernelA<<<2,256,0>>>(d_data, d_result);
// 	// KernelB<<<2,256,0>>>(d_data, d_result);
// 	// KernelC<<<2,256,0>>>(d_data, d_result);
// 	// KernelD<<<2,256,0>>>(d_data, d_result);
// 	KernelE<<<2,256,0>>>(d_data, d_result);

// 	cudaMemcpy(h_result, d_result, sizeof(float) * dataSize, cudaMemcpyDeviceToHost);

// 	free(h_data);
// 	free(h_result);
// 	cudaFree(d_data);
// 	cudaFree(d_result);

// 	return 0;
// }
