#ifndef NOISE_REMOVAL_H
#define NOISE_REMOVAL_H

#include <cuda_runtime.h>

// CUDA kernel declaration
__global__ void gaussianFilterKernel(unsigned char *input, unsigned char *output, int width, int height, int channels);

// Helper function for Gaussian filtering on host
void denoiseImageGPU(const unsigned char *input, unsigned char *output, int width, int height, int channels);

#endif
