/*-----------
 *
 * increment_kernel.cu
 *
 * This is the source file of an increment kernel.
 *
 * This kernel is from CUDA samples. asyncAPI.cu
 *
 * streamsOptBenchmark/increment_kernel.cu
 *
 * By Hao Li
 *
 *------------
 */

 // #include "functions.h"
 // #include <cuda_runtime.h>

 __global__ void increment_kernel(float *g_idata, float *g_odata, int inc_value)
{
	for(int l = 0; l < 100; l++)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // g_data[idx] += inc_values;
    g_odata[idx] = g_idata[idx];
    for(int i = 0; i <= inc_value; ++i){
    	g_odata[idx] += 1;
    }
	}
}

// int main(int argc, char **argv){
// 	int *h_data;

// }
