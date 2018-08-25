/*-----------
 *
 * layerTransform.cu
 *
 * This is the source file of a kernel to transform a layer of a layered 2D texture.
 *
 * This kernel is from CUDA samples. simpleLayeredTexture.cu
 *
 * streamsOptBenchmark/layerTransform.cu
 *
 * By Hao Li
 *
 *------------
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
// declare texture reference for layered 2D float texture
// Note: The "dim" field in the texture reference template is now deprecated.
// Instead, please use a texture type macro such as cudaTextureType1D, etc.

texture<float, cudaTextureType2DLayered> tex;

////////////////////////////////////////////////////////////////////////////////
//! Transform a layer of a layered 2D texture using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

 __global__ void LayerTransformKernel(float *g_idata, float *g_odata, int width, int height, int layer)
{
    for(int l = 0; l < 1000000; l++)
    {
    for(int i = 0; i < layer; i++){
        // calculate this thread's data point
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

        // 0.5f offset and division are necessary to access the original data points
        // in the texture (such that bilinear interpolation will not be activated).
        // For details, see also CUDA Programming Guide, Appendix D
        float u = (x+0.5f) / (float) width;
        float v = (y+0.5f) / (float) height;

        g_odata[i*width*height + y*width + x] = g_idata[i*width*height + y*width + x];

        // read from texture, do expected transformation and write to global memory
        // g_odata[layer*width*height + y*width + x] = -tex2DLayered(tex, u, v, layer) + layer;
        g_odata[i*width*height + y*width + x] = -tex2DLayered(tex, u, v, i) + i;
    }
    }
}

// int main(int argc, char **argv)
// {
//     // generate input data for layered texture
//     unsigned int width=512, height=512, num_layers = 5;
//     unsigned int size = width * height * num_layers * sizeof(float);
//     float *h_data = (float *) malloc(size);

//     for (unsigned int layer = 0; layer < num_layers; layer++)
//         for (int i = 0; i < (int)(width * height); i++)
//         {
//             h_data[layer*width*height + i] = (float)i;
//         }

//     // this is the expected transformation of the input data (the expected output)
//     float *h_data_ref = (float *) malloc(size);

//     for (unsigned int layer = 0; layer < num_layers; layer++)
//         for (int i = 0; i < (int)(width * height); i++)
//         {
//             h_data_ref[layer*width*height + i] = -h_data[layer*width*height + i] + layer;
//         }

//     // allocate device memory for result
//     float *d_data = NULL;
//     (cudaMalloc((void **) &d_data, size));

//     // allocate array and copy image data
//     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//     cudaArray *cu_3darray;
//     (cudaMalloc3DArray(&cu_3darray, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered));
//     cudaMemcpy3DParms myparms = {0};
//     myparms.srcPos = make_cudaPos(0,0,0);
//     myparms.dstPos = make_cudaPos(0,0,0);
//     myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
//     myparms.dstArray = cu_3darray;
//     myparms.extent = make_cudaExtent(width, height, num_layers);
//     myparms.kind = cudaMemcpyHostToDevice;
//     (cudaMemcpy3D(&myparms));

//     // set texture parameters
//     tex.addressMode[0] = cudaAddressModeWrap;
//     tex.addressMode[1] = cudaAddressModeWrap;
//     tex.filterMode = cudaFilterModeLinear;
//     tex.normalized = true;  // access with normalized texture coordinates

//     // Bind the array to the texture
//     (cudaBindTextureToArray(tex, cu_3darray, channelDesc));

//     dim3 dimBlock(8, 8, 1);
//     dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

//     printf("Covering 2D data array of %d x %d: Grid size is %d x %d, each block has 8 x 8 threads\n",
//            width, height, dimGrid.x, dimGrid.y);

//     transformKernel<<< dimGrid, dimBlock >>>(d_data, width, height, 0);  // warmup (for better timing)

//     (cudaDeviceSynchronize());

//     // execute the kernel
//     for (unsigned int layer = 0; layer < num_layers; layer++)
//         transformKernel<<< dimGrid, dimBlock, 0 >>>(d_data, width, height, layer);

//     (cudaDeviceSynchronize());

//     // allocate mem for the result on host side
//     float *h_odata = (float *) malloc(size);
//     // copy result from device to host
//     (cudaMemcpy(h_odata, d_data, size, cudaMemcpyDeviceToHost));

//     // cleanup memory
//     free(h_data);
//     free(h_data_ref);
//     free(h_odata);

//     (cudaFree(d_data));
//     (cudaFreeArray(cu_3darray));

//     // cudaDeviceReset causes the driver to clean up all state. While
//     // not mandatory in normal operation, it is good practice.  It is also
//     // needed to ensure correct operation when the application is being
//     // profiled. Calling cudaDeviceReset causes all profile data to be
//     // flushed before the application exits
//     cudaDeviceReset();

//     return 0;
// }
