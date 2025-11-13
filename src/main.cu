#include "../include/noise_removal.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

int main() {
    std::string inputFolder = "data/input";
    std::string outputFolder = "data/output";

    fs::create_directories(outputFolder);

    int count = 0;

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (!entry.is_regular_file()) continue;

        std::string filePath = entry.path().string();
        int width, height, channels;

        // Load PNG/JPEG
        unsigned char* input = stbi_load(filePath.c_str(), &width, &height, &channels, 3);
        if (!input) {
            std::cout << "Failed to load: " << filePath << std::endl;
            continue;
        }
        channels = 3; // force RGB

        std::vector<unsigned char> output(width * height * channels);

        // Denoise on GPU
        denoiseImageGPU(input, output.data(), width, height, channels);

        // Save output as PNG
        std::string outputPath = outputFolder + "/" + entry.path().filename().string();
        stbi_write_png(outputPath.c_str(), width, height, channels, output.data(), width * channels);

        stbi_image_free(input);

        std::cout << "Processed: " << entry.path().filename() << std::endl;
        count++;
    }

    std::cout << "\nProcessed " << count << " images on GPU!\n";
    return 0;
}
