#include <fstream>
#include <iostream>
using namespace std;
#include "../inc/CIFAR10.hpp"
#ifdef OPENCL
#include "../inc/GPU.hpp"
extern GPU gpu;
#endif

bool readCIFAR10(MNISTImages &images, bool Test)
{
    const string path = "CIFAR10"s;
    auto filename = "data_batch_"s;
    auto noItems = 50000;

    if (Test)
    {
        // 10000 testing images
        noItems = 10000;
        filename = "test_batch"s;
    }

    images.noItems = noItems;
    images.rows = 32;
    images.cols = 32;
    images.noChannels = 3;
    const auto imageSize = size_t(images.rows) * size_t(images.cols) * images.noChannels;
    auto imageData = new char[size_t(noItems) * imageSize];

    auto readImages = 0;
    char dummy;
    for (auto i = 1; i <= noItems / 10000; ++i) //* 分次讀取
    {
        auto f = path + "/" + filename + to_string(i) + ".bin";
        if (Test)
            f = path + "/" + filename + ".bin";
        ifstream inp(f, ios_base::in | ios_base::binary);
        if (!inp)
        {
            std::cerr << "Error opening file: " << f;
            return false;
        }
        for (auto no = 0; no < 10000; ++no)
        {
            inp.read(&dummy, 1);
            inp.read(imageData + readImages * imageSize, imageSize);
            ++readImages;
        }
        inp.close();
    }

#ifndef OPENCL
    images.imageData = new uint8_t[noItems * imageSize];
    for (ulong j = 0; j < imageSize * size_t(noItems); j++)
    {
        images.imageData[j] = uint8_t(imageData[j]);
        // cout << int(images.imageData[j]) << " ";
    }

// cout << "\nPlease maintain '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
#else
    images.imageData = cl::Buffer(gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageSize * noItems, imageData);
    images.normalizedImageData = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, imageSize * noItems * sizeof(float));
    // cout << "\nPlease maintain '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
#endif

    return true;
}
