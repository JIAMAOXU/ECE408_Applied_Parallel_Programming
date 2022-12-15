// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
/*
Cast the image from float to unsigned char
Implement a kernel that casts the image from float * to unsigned char *.
for ii from 0 to (width * height * channels) do
	ucharImage[ii] = (unsigned char) (255 * inputImage[ii])
end
*/
__global__ void cast_to_char(float* input, unsigned char* output, int size){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx < size)
    output[idx] = (uint8_t) ((HISTOGRAM_LENGTH - 1) * input[idx]);
}

/*
Convert the image from RGB to GrayScale
Implement a kernel that converts the RGB image to GrayScale. 
for ii from 0 to height do
	for jj from 0 to width do
		idx = ii * width + jj
		# here channels is 3
		r = ucharImage[3*idx]
		g = ucharImage[3*idx + 1]
		b = ucharImage[3*idx + 2]
		grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b)
	end
end
*/
__global__ void rgb_to_gray(unsigned char* input, unsigned char* ouput, int size){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int r,g,b;
  if(idx < size){
    r = input[idx*3];
    g = input[idx*3 + 1];
    b = input[idx*3 + 2];
    ouput[idx] = (unsigned char) (0.21*r+0.71*g+0.07*b);
  }
}

/*
Compute the histogram of grayImage
Implement a kernel that computes the histogram (like in the lectures) of the image. 
__global__
void histo_kernel(unsigned char *buffer,
long size, unsigned int *histo) {
__shared__ unsigned int histo_private[256];
// warning: this will not work correctly if there are fewer than 256 threads!
if (threadIdx.x < 256)
histo_private[threadIdx.x] = 0;
__syncthreads();
int i = threadIdx.x + blockIdx.x * blockDim.x;
// stride is total number of threads
int stride = blockDim.x * gridDim.x;
while (i < size) {
atomicAdd( &(private_histo[buffer[i]), 1);
i += stride;
}
42
Build Final Histogram
// wait for all other threads in the block to finish
__syncthreads();
if (threadIdx.x < 256)
atomicAdd( &(histo[threadIdx.x]),
private_histo[threadIdx.x] );
}
*/
__global__ void compute_hist(unsigned char* input, unsigned int* output, int Size){
  __shared__ unsigned int hist_private[HISTOGRAM_LENGTH];
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(threadIdx.x < 256) {
    hist_private[threadIdx.x] = 0;
    }
  __syncthreads();
  if(idx < Size) { 
    atomicAdd(&hist_private[input[idx]], 1);
  }
  __syncthreads();
  if(threadIdx.x < 256){
    atomicAdd(&output[threadIdx.x], hist_private[threadIdx.x]);
  }
}

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
__global__ void scan(unsigned int* input, float* output, int size){
  unsigned int stride;
  __shared__ float XY[256*2];
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i < 256){
    XY[threadIdx.x] = input[i];
  }
  for(stride = 1; stride <= 256; stride *= 2){
  __syncthreads();
    unsigned int index = ((1 + threadIdx.x)*2*stride) - 1;
    if(index < 256 ){
      XY[index] += XY[index - stride];
    }
  }
  for(stride = 256 / 2; stride > 0; stride/=2){
  __syncthreads();
    unsigned int index = ((1 + threadIdx.x)*2*stride) - 1;
    if(index + stride < 256 ){
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if(i < 256){
    output[i] = XY[threadIdx.x] / size;
  }
}

/*
Define the histogram equalization function
The histogram equalization function (correct) remaps the cdf of the histogram 
of the image to a linear function and is defined as

def correct_color(val)
	return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
end

def clamp(x, start, end)
	return min(max(x, start), end)
end
*/
__global__ void equalization_function(unsigned char* input, float* output, float* cdf, int size){
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx < size){
    float clamp = 255 * (cdf[input[idx]] - cdf[0]) / (1 - cdf[0]) / (256 - 1);
    output[idx] = (float) min(max(clamp, 0.0), 255.0);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *input;
  float* cdf;
  float* output;
  unsigned char *unsignedcharimg;
  unsigned char *grayimg;
  unsigned int *hist;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **)&input, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&unsignedcharimg, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&grayimg, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&hist, 256 * sizeof(unsigned int));
  cudaMalloc((void **)&cdf, 256 * sizeof(float));
  cudaMalloc((void **)&output, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMemset((void *) hist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *) cdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(input, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  

  //@@ insert code here
  
  dim3 grid (((imageWidth*imageHeight*imageChannels) - 1)/256 + 1);
  dim3 block (256);
  cast_to_char<<<grid, block>>>(input, unsignedcharimg, imageWidth * imageHeight * imageChannels);
  rgb_to_gray<<<grid, block>>>(unsignedcharimg, grayimg, imageWidth * imageHeight);
  compute_hist<<<grid, block>>>(grayimg, hist, imageWidth * imageHeight);
  scan<<<grid, block>>>(hist, cdf, imageWidth * imageHeight);
  equalization_function<<<grid, block>>>(unsignedcharimg, output, cdf, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, output, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(input);
  cudaFree(unsignedcharimg);
  cudaFree(grayimg);
  cudaFree(hist);
  cudaFree(cdf);
  cudaFree(output);

  wbSolution(args, outputImage);
  return 0;
}
