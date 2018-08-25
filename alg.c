#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TVAL(t)             ((t).tv_sec * 1000.0 + (t).tv_usec / 1000.0)
#define TIME_DIFF(t1, t2)   (TVAL(t2) - TVAL(t1))

typedef struct kernel_info
{
  int   reg;    // register number
  int   sharedMem;  // shared memory
  int   dataSize;
  int  BlockDim; // threads
  int  GridDim;  // block number
  int   times;
} Kernel_info;

int intCeil(int a, int b){
  if( (a % b) == 0 ){
    return (a / b);
  }else{
    return (a / b + 1);
  }
}

int main(){

  Kernel_info Kernel[48];
  struct timeval t_start, t_end;

  gettimeofday(&t_start, NULL);

  for(int i; i < 48; i++){
    Kernel[i].reg = (i+1);
    Kernel[i].sharedMem = (i+1)*4;
    Kernel[i].dataSize = (9-i)*8;
  }

  int N = 48;
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
          Kernel[n].GridDim = b;
          Kernel[n].BlockDim = intCeil(Kernel[n].dataSize, Kernel[n].GridDim);
        }
      }
      if(V[n][b][W]<V[n][b-1][W]){
        B = B-b;
        W = W - intCeil(Kernel[n].dataSize, b*32);
      }
    }
  }

  gettimeofday(&t_end, NULL);
  printf("Total time taken: %lf ms\n", TIME_DIFF(t_start, t_end));

  return 0;
}