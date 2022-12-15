#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 3
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define MASK_SIZE (TILE_WIDTH + (MASK_WIDTH - 1))
//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
 __shared__ float N_ds[MASK_SIZE][MASK_SIZE][MASK_SIZE];

  // Shifting from output coordinates to input coordinates 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int x_col_o = bx * TILE_WIDTH + tx;
  int y_row_o = by * TILE_WIDTH + ty;
  int z_height_o = bz * TILE_WIDTH + tz;

  int x_col_i = x_col_o - MASK_RADIUS;
  int y_row_i = y_row_o - MASK_RADIUS;
  int z_height_i = z_height_o - MASK_RADIUS;

// __shared__ float N_ds[TILE_SIZE+MAX_MASK_WIDTH-1][TILE_SIZE+MAX_MASK_HEIGHT-1];
// If ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
//  N_ds[ty][tx] = data[row_i * pitch + col_i];
// } else{
//  N_ds[ty][tx] = 0.0f;
// }

// Taking Care of Boundaries
if (
      (x_col_i >= 0)    &&  (x_col_i < x_size)    &&
      (y_row_i >= 0)    &&  (y_row_i < y_size)    &&
      (z_height_i >= 0) &&  (z_height_i < z_size)
    )
    {N_ds[tz][ty][tx] = input[z_height_i * y_size * x_size + y_row_i * x_size + x_col_i];}
else
    {N_ds[tz][ty][tx] = 0.0f;}

__syncthreads(); // wait for tile

// Not All Threads Calculate Output
  float Pvalue = 0.0f;
  if(tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH ){
    for(int i = 0; i < MASK_WIDTH; i++) { 
      for(int j = 0; j < MASK_WIDTH; j++) {
        for (int k = 0; k < MASK_WIDTH; k++) {
          Pvalue += Mc[i][j][k] * N_ds[i+tz][j+ty][k+tx];  
        }
      }
    }
    if(x_col_o < x_size && y_row_o < y_size &&  z_height_o < z_size) {
      output[z_height_o * y_size * x_size + y_row_o * x_size + x_col_o] = Pvalue; 
    }
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput,(inputLength - 3)*sizeof(float));
  cudaMalloc((void**) &deviceOutput,(inputLength - 3)*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");


  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");


  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(((float)x_size) / TILE_WIDTH), 
               ceil(((float)y_size) / TILE_WIDTH), 
               ceil(((float)z_size) / TILE_WIDTH));
  dim3 DimBlock(MASK_SIZE, MASK_SIZE, MASK_SIZE);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
   cudaMemcpy(hostOutput+3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
