/* Udacity Homework 3
   HDR Tone-mapping

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

/**
* histogram
*/

__global__ void histogram(unsigned int *d_bins,
                          const float *d_in,
                          const float lumMin,
                          const float lumRange,
                          int BIN_COUNT)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int bin = (d_in[i] - lumMin) / lumRange * BIN_COUNT;
    atomicAdd(&(d_bins[bin]), 1);
}

/**
* reduce max
*/
__global__
void reduce_max_kernel(float * d_out, const float * d_in)
{
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
      if (tid < s)
      {
          sdata[tid] = max(sdata[tid + s], sdata[tid]);
      }
      __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
      d_out[blockIdx.x] = sdata[0];
  }
}

void reduce_max(float &max_logLum, const float * d_in, const size_t size)
{
  float *d_intermediate;
  float *d_out;
  cudaMalloc((void **) &d_intermediate, size * sizeof(float)); // overallocated
  cudaMalloc((void **) &d_out, sizeof(float)); // overallocated

  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks = size / maxThreadsPerBlock;

  reduce_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);

  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;

  reduce_max_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);

  cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  printf("maximum %f \n", max_logLum);
}

/**
* reduce min
*/

__global__
void reduce_min_kernel(float * d_out, const float * d_in)
{
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
      if (tid < s)
      {
          sdata[tid] = min(sdata[tid + s], sdata[tid]);
      }
      __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
      d_out[blockIdx.x] = sdata[0];
  }
}

void reduce_min(float &max_logLum, const float * d_in, const size_t size)
{
  float *d_intermediate;
  float *d_out;
  checkCudaErrors(cudaMalloc((void **) &d_intermediate, size * sizeof(float))); // overallocated
  checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(float))); // overallocated

  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks = size / maxThreadsPerBlock;

  reduce_min_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);

  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;

  reduce_min_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);

  checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  printf("minimum %f \n", max_logLum);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
  */
  size_t size = numRows * numCols;
  reduce_max(max_logLum, d_logLuminance, size);
  reduce_min(min_logLum, d_logLuminance, size);
  /*
    2) subtract them to find the range
  */
  float logLumRange = max_logLum - min_logLum;
  /*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  */
  unsigned int *d_histo;
  const int maxThreadsPerBlock = 1024;
  checkCudaErrors(cudaMalloc((void **) &d_histo, numBins * sizeof(int)));
  checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(int)));
  histogram<<<size / maxThreadsPerBlock, maxThreadsPerBlock>>>(d_histo,
                                                               d_logLuminance,
                                                               min_logLum,
                                                               logLumRange,
                                                               numBins);

  /*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)
  */

  

}
