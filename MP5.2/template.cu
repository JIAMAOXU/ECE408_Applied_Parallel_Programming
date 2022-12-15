// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *S) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

// David Kirk/NVIDIA and Wen-mei Hwu, 2006-2016 
/*
__global__ void Brent_Kung_scan_kernel(float *X, float *Y,
 int InputSize) {

 __shared__ float XY[SECTION_SIZE];
 int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
 if (i < InputSize) XY[threadIdx.x] = X[i];
 if (i+blockDim.x < InputSize) XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
 for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
 __syncthreads();
 int index = (threadIdx.x+1) * 2* stride -1;
 if (index < SECTION_SIZE) {
 XY[index] += XY[index - stride];
 }
 }

 for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
 __syncthreads();
 int index = (threadIdx.x+1)*stride*2 - 1;
 if(index + stride < SECTION_SIZE) {
 XY[index + stride] += XY[index];
 }
 }

 __syncthreads();
 if (i < InputSize) Y[i] = XY[threadIdx.x];
 if (i+blockDim.x < InputSize) Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];
 }

*/

// Referenced from textbook shown above//
__shared__ float XY[2*BLOCK_SIZE];
  int index = 2*blockIdx.x*blockDim.x + threadIdx.x;
  if (index < len) {XY[threadIdx.x] = input[index];} 
  if (index+blockDim.x < len) {XY[threadIdx.x+blockDim.x] = input[index+blockDim.x];} 
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int Index = (threadIdx.x+1)*2*stride - 1;
    if (Index < 2*BLOCK_SIZE) {
      XY[Index] += XY[Index - stride];
    }
  }

  for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
    __syncthreads();
    int Index = (threadIdx.x+1)*stride*2 - 1;
    if (Index + stride < 2*BLOCK_SIZE) {
      XY[Index+stride] += XY[Index];
    }
  }

  __syncthreads();
  if(index < len)
    output[index] = XY[threadIdx.x];
  if (index+blockDim.x <len) 
    output[index+blockDim.x] = XY[threadIdx.x + blockDim.x];

/*
We need to add one more parameter S, which has the dimension of
InputSize/SECTION_SIZE. At the end of the kernel, we add a conditional statement for the
last thread in the block to write the output value of the last XY element in the scan block to
the blockIdx.x position of S

 __syncthreads();
 if (threadIdx.x == blockDim.x-1) {
 S[blockIdx.x] = XY[SECTION_SIZE – 1];
 }

*/
  __syncthreads();
  if (threadIdx.x == BLOCK_SIZE-1) {
    S[blockIdx.x] = XY[2*BLOCK_SIZE-1];
  }
}

__global__ void helperfunction(float *S, float *sum, int len) {
  if(len > 2 * blockIdx.x * blockDim.x + threadIdx.x + blockDim.x ){
    if (blockIdx.x > 0 ) {
      sum[(2*blockIdx.x * blockDim.x) + threadIdx.x + blockDim.x] = sum[(2*blockIdx.x * blockDim.x) + threadIdx.x + blockDim.x] + S[blockIdx.x - 1];
    } 
 }
  if(len > 2*blockIdx.x*blockDim.x + threadIdx.x ){
    if (blockIdx.x > 0 ) {
      sum[(2*blockIdx.x*blockDim.x) + threadIdx.x] = sum[(2*blockIdx.x*blockDim.x) + threadIdx.x] + S[blockIdx.x - 1];
  }
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSum, ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(2.0*BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, deviceSum);
  scan<<<1, dimBlock>>>(deviceSum, deviceSum, ceil(numElements/(2.0*BLOCK_SIZE)), deviceInput);
  helperfunction<<<dimGrid, dimBlock>>>(deviceSum, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
