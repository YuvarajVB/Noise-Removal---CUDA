# GPU Noise Removal 

This project implements GPU-accelerated image denoising using CUDA. It reads noisy PNG or JPEG images and applies a Gaussian filter to reduce noise, leveraging parallel computation for fast processing. The solution uses the lightweight `stb_image` library for image I/O and does not require OpenCV. Processed images are saved to an output folder, demonstrating efficient GPU-based image processing on multiple images.

## Key Features
- Supports both PNG and JPEG image formats.
- Fully GPU-accelerated denoising with a 3x3 Gaussian filter.
- Lightweight, with no OpenCV dependency.
- Can process multiple images in a folder automatically.
