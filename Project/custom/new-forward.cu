#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

//#define M2
// 2pt+1pt+3pt+4pt
// #define Tiled_shared_memory_convolution
// #define Sweep_parameter
// #define Tuning_with_restrict 
 #define Using_stream_overlap


//------------------------------------------------M2 Code------------------------------------------------//
#ifdef M2
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    /*We will divide the shared memory between input buffer and filter inside the kernel. The first
    X_tile_width* X_tile_width entries are allocated to the input tiles and the rest of the entries
    are allocated to the weight values.*/

    //implmented from the textbook
    /*__global__ void
ConvLayerForward_Kernel(int C, int W_grid, int K, float* X, float* W, float* Y)
{
 int n, m, h0, w0, h_base, w_base, h, w;
 int X_tile_width = TILE_WIDTH + K-1;
 extern __shared__ float shmem[];
 float* X_shared = &shmem[0];
 float* W_shared = &shmem[X_tile_width * X_tile_width];
 n = blockIdx.x;
 m = blockIdx.y;
 h0 = threadIdx.x;
 w0 = threadIdx.y;
 h_base = (blockIdx.z / W_grid) * TILE_SIZE; // vertical base out data index for the block
 w_base = (blockIdx.z % W_grid) * TILE_SIZE; // horizontal base out data index for the block
 h = h_base + h0;
 w = w_base + w0;
 float acc = 0.;
 int c, j, k, p, q;
 for (c = 0; c < C; c++) { // sum over all input channels
 // load weights for W [m, c,..],
// h0 and w0 used as shorthand for threadIdx.x
// and threadIdx.y
 if (( h0 < K) && ( w0 < K))
Third Edition Preproduction Draft
© David Kirk/NVIDIA and Wen-mei Hwu, 2006-2016 16
 W_shared[h0, w0]= W [m, c, h0, w0];
 __syncthreads();
 // load tile from X[n, c,…] into shared memory

 for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
 for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
 X_shared[i - h_base, j - w_base] = X[n, c, h, w]
 }
 __syncthreads();
 for (p = 0; p < K; p++) {
 for (q = 0; q < K; q++)
 acc = acc + X_shared[h + p, w + q] * W_shared[p, q];
 }
 __syncthreads();
 }
 Y[n, m, h, w] = acc;
}*/
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    int H_grid = ceil(1.0*H_out / TILE_WIDTH);

    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;
    int c, p, q;
    for (c = 0; c < Channel; c++) {
        if((h0 < K) && (w0 < K)){
            W_shared[h0 * K + w0] = mask_4d(m, c, h0, w0);
        }

        __syncthreads();  // load tile from X[n, c,…] into shared memory
        for(int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
            for(int j = w; j < w_base + X_tile_width; j += TILE_WIDTH){
                if(i < Height && j < Width) {
                    X_shared[(i - h_base) * X_tile_width + j - w_base] = in_4d(n, c, i, j);
                }
                else {
                    X_shared[(i - h_base) * X_tile_width + j - w_base] = 0;
                }
            }
        }

        __syncthreads();
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++){
                acc += X_shared[(h0 + p) * X_tile_width + w0 + q] * W_shared[p * K + q];
            }
        }
        __syncthreads();
    }

    if(n < Batch && m < Map_out && h < Height_out && w < Width_out){
        out_4d(n, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **)device_input_ptr, (Batch * Channel * Height * Width * sizeof(float)));
    cudaMalloc((void **)device_output_ptr, (Batch * Map_out * Height_out * Width_out * sizeof(float)));
    cudaMalloc((void **)device_mask_ptr, (Channel * Map_out * K * K * sizeof(float)));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float),cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = ceil(Width_out / (float) TILE_WIDTH);
    int H_grid = ceil(Height_out / (float) TILE_WIDTH);
    int Z = H_grid * W_grid;
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Batch, Map_out, Z);
    size_t size = ((TILE_WIDTH+K-1) * (TILE_WIDTH+K-1)+K*K)*sizeof(float);
    
    conv_forward_kernel<<<dimGrid, dimBlock, size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int temp = (Batch * Map_out * Height_out * Width_out) * sizeof(float);
    cudaMemcpy(host_output, device_output, temp, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}
#endif

//------------------------------------------------Optimization 1 Code------------------------------------------------//
#ifdef Tiled_shared_memory_convolution
__constant__ float mem[6000];
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    extern __shared__ float shared_mem[];
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = (Width_out - 1)/TILE_WIDTH + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mem[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shared_mem(i2, i1, i0) shared_mem[(i2) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)) + (i1) * (TILE_WIDTH + K - 1) + i0]
    
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    int h_base = TILE_WIDTH * (blockIdx.z/W_grid);
    int w_base = TILE_WIDTH * (blockIdx.z%W_grid);
    for (int c = 0; c < Channel; c++){
        for(int i = threadIdx.y; i < (TILE_WIDTH + K - 1); i += TILE_WIDTH){
            for(int j = threadIdx.x; j < (TILE_WIDTH + K - 1); j += TILE_WIDTH){
                if (h_base + i < Height && w_base + j < Width){
                    shared_mem(c, i, j) = in_4d(blockIdx.x, c, h_base + i, w_base + j);
                }   
            }
        }
    }

    __syncthreads();

    if (h < Height_out && w < Width_out){
        float acc = 0;
        for (int c = 0; c < Channel; c++){
            for (int p = 0; p < K; p++){
                for (int q = 0; q < K; q++){
                    acc += shared_mem(c, threadIdx.y + p, threadIdx.x + q) * mask_4d(blockIdx.y, c, p, q);
                }
            }
        }
        out_4d(m, blockIdx.y, h, w) = acc;
    }
#undef out_4d
#undef in_4d
#undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **)device_input_ptr, (Batch * Channel * Height * Width * sizeof(float)));
    cudaMalloc((void **)device_output_ptr, (Batch * Map_out * Height_out * Width_out * sizeof(float)));
    cudaMalloc((void **)device_mask_ptr, (Channel * Map_out * K * K * sizeof(float)));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mem, host_mask, (Map_out * Channel * K * K)*sizeof(float));

}



__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int H_grid = ceil((float)(Height_out)/TILE_WIDTH);
    int W_grid = ceil((float)(Width_out)/TILE_WIDTH);
    int Z = W_grid * H_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Batch, Map_out, Z);
    size_t size = (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) * sizeof(float);
    conv_forward_kernel<<<dimGrid, dimBlock, Channel*size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int temp = (Batch * Map_out * Height_out * Width_out) * sizeof(float);
    cudaMemcpy(host_output, device_output, temp, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

#endif


//------------------------------------------------Optimization 6 Code------------------------------------------------//
#ifdef Sweep_parameter

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = ceil((float)Width_out/TILE_WIDTH);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int m = blockIdx.x;
    int b = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    if((h < Height_out) && (w < Width_out)){
        for(int c = 0; c < Channel; c++){
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++){
                if(!( w + q > Width || h+p > Height)) 
                        acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
        }
    }
}
        out_4d(b, m, h, w) = acc;
    }
#undef out_4d
#undef in_4d
#undef mask_4d
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **)device_input_ptr, (Batch * Channel * Height * Width * sizeof(float)));
    cudaMalloc((void **)device_output_ptr, (Batch * Map_out * Height_out * Width_out * sizeof(float)));
    cudaMalloc((void**)device_mask_ptr, sizeof(float) * Map_out * Channel * K * K);    
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);

}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    int W_grid = ceil((float)(Width)/TILE_WIDTH);
    int H_grid = ceil((float)(Height)/TILE_WIDTH);
    int Z = W_grid * H_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Map_out,Batch, Z);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int temp = (Batch * Map_out * Height_out * Width_out) * sizeof(float);
    cudaMemcpy(host_output, device_output, temp, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}
#endif


//------------------------------------------------Optimization 5 Code------------------------------------------------//
#ifdef Tuning_with_restrict
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
//     if((h < Height_out) && (w < Width_out)){
//         for(int c = 0; c < Channel; c++){
//             for(int p = 0; p < K; p++){
//                 for(int q = 0; q < K; q++){
//                         acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
//         }
//     }
// }
//         out_4d(b, m, h, w) = acc;
//     }

    // unrolling process
    if((h < Height_out) && (w < Width_out)){
        for(int c = 0; c < Channel; c++){
            acc += in_4d(b,c,h+0,w+0) * mask_4d(m,c,0,0);
            acc += in_4d(b,c,h+0,w+1) * mask_4d(m,c,0,1);
            acc += in_4d(b,c,h+0,w+2) * mask_4d(m,c,0,2);
            acc += in_4d(b,c,h+0,w+3) * mask_4d(m,c,0,3);
            acc += in_4d(b,c,h+0,w+4) * mask_4d(m,c,0,4);
            acc += in_4d(b,c,h+0,w+5) * mask_4d(m,c,0,5);
            acc += in_4d(b,c,h+0,w+6) * mask_4d(m,c,0,6);

            acc += in_4d(b,c,h+1,w+0) * mask_4d(m,c,1,0);
            acc += in_4d(b,c,h+1,w+1) * mask_4d(m,c,1,1);
            acc += in_4d(b,c,h+1,w+2) * mask_4d(m,c,1,2);
            acc += in_4d(b,c,h+1,w+3) * mask_4d(m,c,1,3);
            acc += in_4d(b,c,h+1,w+4) * mask_4d(m,c,1,4);
            acc += in_4d(b,c,h+1,w+5) * mask_4d(m,c,1,5);
            acc += in_4d(b,c,h+1,w+6) * mask_4d(m,c,1,6);

            acc += in_4d(b,c,h+2,w+0) * mask_4d(m,c,2,0);
            acc += in_4d(b,c,h+2,w+1) * mask_4d(m,c,2,1);
            acc += in_4d(b,c,h+2,w+2) * mask_4d(m,c,2,2);
            acc += in_4d(b,c,h+2,w+3) * mask_4d(m,c,2,3);
            acc += in_4d(b,c,h+2,w+4) * mask_4d(m,c,2,4);
            acc += in_4d(b,c,h+2,w+5) * mask_4d(m,c,2,5);
            acc += in_4d(b,c,h+2,w+6) * mask_4d(m,c,2,6);  

            acc += in_4d(b,c,h+3,w+0) * mask_4d(m,c,3,0);
            acc += in_4d(b,c,h+3,w+1) * mask_4d(m,c,3,1);
            acc += in_4d(b,c,h+3,w+2) * mask_4d(m,c,3,2);
            acc += in_4d(b,c,h+3,w+3) * mask_4d(m,c,3,3);
            acc += in_4d(b,c,h+3,w+4) * mask_4d(m,c,3,4);
            acc += in_4d(b,c,h+3,w+5) * mask_4d(m,c,3,5);
            acc += in_4d(b,c,h+3,w+6) * mask_4d(m,c,3,6);

            acc += in_4d(b,c,h+4,w+0) * mask_4d(m,c,4,0);
            acc += in_4d(b,c,h+4,w+1) * mask_4d(m,c,4,1);
            acc += in_4d(b,c,h+4,w+2) * mask_4d(m,c,4,2);
            acc += in_4d(b,c,h+4,w+3) * mask_4d(m,c,4,3);
            acc += in_4d(b,c,h+4,w+4) * mask_4d(m,c,4,4);
            acc += in_4d(b,c,h+4,w+5) * mask_4d(m,c,4,5);
            acc += in_4d(b,c,h+4,w+6) * mask_4d(m,c,4,6);

            acc += in_4d(b,c,h+5,w+0) * mask_4d(m,c,5,0);
            acc += in_4d(b,c,h+5,w+1) * mask_4d(m,c,5,1);
            acc += in_4d(b,c,h+5,w+2) * mask_4d(m,c,5,2);
            acc += in_4d(b,c,h+5,w+3) * mask_4d(m,c,5,3);
            acc += in_4d(b,c,h+5,w+4) * mask_4d(m,c,5,4);
            acc += in_4d(b,c,h+5,w+5) * mask_4d(m,c,5,5);
            acc += in_4d(b,c,h+5,w+6) * mask_4d(m,c,5,6);

            acc += in_4d(b,c,h+6,w+0) * mask_4d(m,c,6,0);
            acc += in_4d(b,c,h+6,w+1) * mask_4d(m,c,6,1);
            acc += in_4d(b,c,h+6,w+2) * mask_4d(m,c,6,2);
            acc += in_4d(b,c,h+6,w+3) * mask_4d(m,c,6,3);
            acc += in_4d(b,c,h+6,w+4) * mask_4d(m,c,6,4);
            acc += in_4d(b,c,h+6,w+5) * mask_4d(m,c,6,5);
            acc += in_4d(b,c,h+6,w+6) * mask_4d(m,c,6,6);
        }
        out_4d(b, m, h, w) = acc;
    }
#undef out_4d
#undef in_4d
#undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **)device_input_ptr, (Batch * Channel * Height * Width * sizeof(float)));
    cudaMalloc((void **)device_output_ptr, (Batch * Map_out * Height_out * Width_out * sizeof(float)));
    cudaMalloc((void **)device_mask_ptr, (Channel * Map_out * K * K * sizeof(float)));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float),cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Map_out, Y, Batch);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, 
                                                Batch, Map_out, Channel, 
                                                Height, Width, K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int temp = (Batch * Map_out * Height_out * Width_out) * sizeof(float);
    cudaMemcpy(host_output, device_output, temp, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}
#endif

//------------------------------------------------Optimization 11 Code------------------------------------------------//
#ifdef Using_stream_overlap
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    const int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    if((h < Height_out) && (w < Width_out)){
        for(int c = 0; c < Channel; c++){
            for(int p = 0; p < K; p++)
                for(int q = 0; q < K; q++)
                        acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
        }
        out_4d(b, m, h, w) = acc;
    }
#undef out_4d
#undef in_4d
#undef mask_4d    
}
// reference : https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/ 



__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int seg_size = 10;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int segX = Batch * Channel * Height * Width;
    int segY = Batch * Map_out * Height_out * Width_out; 

    int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;


    cudaMalloc((void **)device_input_ptr, (Batch * Channel * Height * Width * sizeof(float)));
    cudaMalloc((void **)device_output_ptr, (Batch * Map_out * Height_out * Width_out * sizeof(float)));
    cudaMalloc((void **)device_mask_ptr, (Channel * Map_out * K * K * sizeof(float)));


    cudaStream_t stream[seg_size];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[2]);
    cudaStreamCreate(&stream[3]);
    cudaStreamCreate(&stream[4]);
    cudaStreamCreate(&stream[5]);
    cudaStreamCreate(&stream[6]);
    cudaStreamCreate(&stream[7]);
    cudaStreamCreate(&stream[8]);
    cudaStreamCreate(&stream[9]);

    cudaMemcpyAsync(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 0), host_input + ((segX / seg_size) * 0), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 1), host_input + ((segX / seg_size) * 1), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 2), host_input + ((segX / seg_size) * 2), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[2]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 3), host_input + ((segX / seg_size) * 3), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[3]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 4), host_input + ((segX / seg_size) * 4), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[4]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 5), host_input + ((segX / seg_size) * 5), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[5]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 6), host_input + ((segX / seg_size) * 6), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[6]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 7), host_input + ((segX / seg_size) * 7), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[7]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 8), host_input + ((segX / seg_size) * 8), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[8]);
    cudaMemcpyAsync((*device_input_ptr) + ((segX / seg_size) * 9), host_input + ((segX / seg_size) * 9), (segX / seg_size) * sizeof(float), cudaMemcpyHostToDevice, stream[9]);


    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(Map_out, Y, Batch/seg_size);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[0]>>>((*device_output_ptr) + ((segY / seg_size) * 0), (*device_input_ptr) + ((segX / seg_size) * 0), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[1]>>>((*device_output_ptr) + ((segY / seg_size) * 1), (*device_input_ptr) + ((segX / seg_size) * 1), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[2]>>>((*device_output_ptr) + ((segY / seg_size) * 2), (*device_input_ptr) + ((segX / seg_size) * 2), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[3]>>>((*device_output_ptr) + ((segY / seg_size) * 3), (*device_input_ptr) + ((segX / seg_size) * 3), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[4]>>>((*device_output_ptr) + ((segY / seg_size) * 4), (*device_input_ptr) + ((segX / seg_size) * 4), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[5]>>>((*device_output_ptr) + ((segY / seg_size) * 5), (*device_input_ptr) + ((segX / seg_size) * 5), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[6]>>>((*device_output_ptr) + ((segY / seg_size) * 6), (*device_input_ptr) + ((segX / seg_size) * 6), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[7]>>>((*device_output_ptr) + ((segY / seg_size) * 7), (*device_input_ptr) + ((segX / seg_size) * 7), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[8]>>>((*device_output_ptr) + ((segY / seg_size) * 8), (*device_input_ptr) + ((segX / seg_size) * 8), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<dim_grid, dimBlock, 0, stream[9]>>>((*device_output_ptr) + ((segY / seg_size) * 9), (*device_input_ptr) + ((segX / seg_size) * 9), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 0), (*device_output_ptr) + ((segY / seg_size) * 0), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 1), (*device_output_ptr) + ((segY / seg_size) * 1), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 2), (*device_output_ptr) + ((segY / seg_size) * 2), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[2]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 3), (*device_output_ptr) + ((segY / seg_size) * 3), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[3]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 4), (*device_output_ptr) + ((segY / seg_size) * 4), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[4]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 5), (*device_output_ptr) + ((segY / seg_size) * 5), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[5]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 6), (*device_output_ptr) + ((segY / seg_size) * 6), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[6]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 7), (*device_output_ptr) + ((segY / seg_size) * 7), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[7]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 8), (*device_output_ptr) + ((segY / seg_size) * 8), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[8]);
    cudaMemcpyAsync((float*)host_output + ((segY / seg_size) * 9), (*device_output_ptr) + ((segY / seg_size) * 9), (segY / seg_size) * sizeof(float), cudaMemcpyDeviceToHost, stream[9]);

    cudaDeviceSynchronize();


    // Free device memory
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);
}
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
}
#endif

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}