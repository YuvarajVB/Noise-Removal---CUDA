#include "noise_removal.h"
#include <cuda_runtime.h>
#include <iostream>

// Simple 3x3 Gaussian filter kernel on GPU
__global__ void gaussianFilterKernel(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    const int kernelSum = 16;

    for (int c = 0; c < channels; c++)
    {
        int sum = 0;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = min(max(x + dx, 0), width - 1);
                int ny = min(max(y + dy, 0), height - 1);

                sum += input[(ny * width + nx) * channels + c] * kernel[dy + 1][dx + 1];
            }
        }

        output[(y * width + x) * channels + c] = sum / kernelSum;
    }
}

// Host wrapper
void denoiseImageGPU(const unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    unsigned char *d_input, *d_output;
    size_t bytes = width * height * channels;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gaussianFilterKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
