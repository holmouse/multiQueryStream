/*-----------
 *
 * functions.cu
 *
 * This is the source file of non-cuda functions in this benchmark.
 *
 * streamsOptBenchmark/functions.cu
 *
 * By Hao Li
 *
 *------------
 */

#include "functions.cuh"

// // Initialize data of certain size
// void constantInit(float *data, int size)
// {
// 	time_t t;
// 	srand((unsigned) time(&t));
//     for (int i = 0; i < size; ++i)
//     {
//         // data[i] = rand()%(size+1);	// random set data[i] from 0 to size
//         data[i] = ((float)rand()/(float)(RAND_MAX)) * size;
//         // printf("%f\n", data[i]);
//     }
// }

// // Initialize a matrix
// void initMatrix(Matrix &M, int size, bool host)
// {
// 	M.width = MATRIX_SIZE;
// 	M.height = MATRIX_SIZE;
// 	M.stride = M.width;

// 	// if the matrix is on host, using malloc, else using cudaMalloc
// 	if(host == 1){
// 		M.elements = (float *)malloc(size);
// 		constantInit(M.elements, M.width * M.height);
// 	}else{
// 		cudaMalloc((void **) &M.elements, size);
// 	}
// }

// // Choose a kernel to launch
// void lauchKernel(int kernelNum)
// {
// 	switch(kernelNum)
// 	{
// 		// case 0:
// 		// 	kernel_0;
// 		// 	break;
// 		// case 1:
// 		// 	kernel_1;
// 		// 	break;
// 		// case 2:
// 		// 	kernel_2;
// 		// 	break;
// 		// case 3:
// 		// 	kernel_3;
// 		// 	break;
// 		// case 4:
// 		// 	kernel_4;
// 		// 	break;
// 		// case 5:
// 		// 	kernel_5;
// 		// 	break;
// 		// case 6:
// 		// 	kernel_6;
// 		// 	break;
// 		// case 7:
// 		// 	kernel_7;
// 		// 	break;
// 		// case 8:
// 		// 	kernel_8;
// 		// 	break;
// 		// case 9:
// 		// 	kernel_9;
// 		// 	break;
// 		// case 10:
// 		// 	kernel_10;
// 		// 	break;
// 		// case 11:
// 		// 	kernel_11;
// 		// 	break;
// 		// case 12:
// 		// 	kernel_12;
// 		// 	break;
// 		// case 13:
// 		// 	kernel_13;
// 		// 	break;
// 	}
// }

// // Initialize paremeters for each kernel
// void initKernel(Kernel_info *Kernel, bool isDefault, int size)
// {
// 	if(isDefault){
// 		for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
// 			Kernel[i].BlockDim = dim3(DEFAULT_THREADS,1,1);
// 			// Kernel[i].BlockDim.x = DEFAULT_THREADS;
// 			printf("%d\t", Kernel[i].BlockDim.x);
// 			Kernel[i].GridDim = dim3(ceil(size/DEFAULT_THREADS),1,1);
// 			// Kernel[i].GridDim.x = ceil(size/DEFAULT_THREADS);
// 			printf("%d\n", Kernel[i].GridDim.x);
// 			Kernel[i].loopTimes = 1;
// 		}
// 	}else{
// 		// input kernel info one by one
// 		for(int i = 0; i < TOTAL_KERNEL_NUM; i++){
// 			printf("Please input total size of data %d: ", i+1);
// 			scanf("%d", &Kernel[i].dataSize);

// 			int thread = 0;
// 			printf("Please input block size of kernel %d: ", i+1);
// 			scanf("%d", &thread);
// 			Kernel[i].BlockDim = dim3(thread,1,1);
			
// 			// int blockNum = 0;
// 			// printf("Please input block number of kernel %d: ", i+1);
// 			// scanf("%d", &blockNum);
// 			Kernel[i].GridDim = dim3(ceil(size/thread),1,1);
// 		}
// 	}

// }

// int default()
// {

// }

// Based on command line argument to choose different functions to run
int primitiveFunc(int argc, char **argv)
{
	if(argc > 3){
		printf("Command Line Argument:\n\n");
		printf("no prefix: running all the kernels with default block size and block number\n");
		printf("--help\t\t\t\t(-h) \n\t Show user help.\n\n");
		printf("--manual [num]\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
		printf("--benchmark [num]\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n");
		return 1;
	}else if(argc == 1){

		return 0;
	}else if((strcmp(argv[1], "--help")==0) || (strcmp(argv[1], "-h")==0)){
		printf("Command Line Argument:\n\n");
		printf("no prefix: running all the kernels with default block size and block number\n");
		printf("--help\t\t\t\t(-h) \n\t Show user help.\n\n");
		printf("--manual [num]\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
		printf("--benchmark [num]\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n");
		return 0;
	}else if((strcmp(argv[1], "--manual")==0) || (strcmp(argv[1], "-m")==0)){
		if(atoi(argv[2]) < 0 || atoi(argv[2]) > TOTAL_KERNEL_NUM){
			printf("The total number of kernels should be 0~%d\n", TOTAL_KERNEL_NUM);
			return 1;
		}else{

		}

		return 0;
	}else if((strcmp(argv[1], "--benchmark")==0) || (strcmp(argv[1], "-b")==0)){
		if(atoi(argv[2]) < 0 || atoi(argv[2]) > TOTAL_KERNEL_NUM){
			printf("The total number of kernels should be 0~%d\n", TOTAL_KERNEL_NUM);
			return 1;
		}else{

		}

		return 0;
	}else{
		printf("Command Line Argument:\n\n");
		printf("no prefix: running all the kernels with default block size and block number\n");
		printf("--help\t\t\t\t(-h) \n\t Show user help.\n\n");
		printf("--manual [num]\t\t\t(-m [num]) \n\t Running random number of kernels and input each kernel's informantion manually.\n\n");
		printf("--benchmark [num]\t\t(-b [num]) \n\t Running random number of kernels, got the results of both assigning kernel info ramdom and optimized.\n");
		return 1;
	}
}