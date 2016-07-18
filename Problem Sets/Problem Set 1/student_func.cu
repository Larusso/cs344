#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  size_t c = threadIdx.x + blockIdx.x * blockDim.x;
  size_t r = threadIdx.y + blockIdx.y * blockDim.y;

  if(r <= numRows && c <= numCols)
  {
    uchar4 rgba = rgbaImage[r * numCols + c];
    float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
    greyImage[r * numCols + c] = channelSum;
  }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  size_t block_size = 32;
  const dim3 blockSize(block_size, block_size, 1);
  const dim3 gridSize(ceil(double(numCols)/block_size), ceil(double(numRows) / block_size), 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
