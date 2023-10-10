#include "../inc/MNIST.hpp"

#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
#ifdef OPENCL
#include "../inc/GPU.hpp"
extern GPU gpu;
#endif
// the raw data area stored in Big-endian (https://en.wikipedia.org/wiki/Endianness) order,
// but Intel processors use little-endian format in-memory, so we cannot directly
// read the file content and stored it into memory...
static int32_t readInt(istream &inp)
{
    unsigned char buf[4];
    inp.read((char *)buf, 4);
    return buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
}

// Details: http://yann.lecun.com/exdb/mnist/
bool readImages(const string &filename, MNISTImages &images)
{
    ifstream inp(filename, ios_base::in | ios_base::binary);
    auto magic = readInt(inp);
    if (magic != 2051)
        return false;
    auto noImages = readInt(inp); // This is the number of images
    auto rows = readInt(inp);     // number of rows of each image
    auto cols = readInt(inp);     // number of columns of each image

    images.noItems = noImages;
    images.rows = rows;
    images.cols = cols;
    images.noChannels = 1;

    // raw image data
    auto imageData = new char[size_t(noImages) * size_t(rows) * size_t(cols)];
    inp.read(imageData, size_t(noImages) * size_t(rows) * size_t(cols));
    inp.close();

#ifndef OPENCL

    auto imagesize = images.noItems * images.rows * images.cols;
    images.imageData = new uint8_t[imagesize];
    for (int i = 0; i < imagesize; i++)
    {
        images.imageData[i] = uint8_t(imageData[i]);
    }
    // cout << "\nPlease maintain '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
#else

    images.imageData = cl::Buffer(gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, images.rows * images.cols * images.noChannels * noImages, imageData);
    images.normalizedImageData = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, images.rows * images.cols * images.noChannels * noImages * sizeof(float));
    // cout << "\nPlease maintain '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
#endif

    return true;
}

// This function de-allocate memory for images in MNIST dataset.
void destroyMNIST(MNISTImages &img)
{
#ifndef OPENCL
    delete[] img.imageData;
    delete[] img.normalizedImageData;
#else
    img.imageData = cl::Buffer();
    img.normalizedImageData = cl::Buffer();
#endif
}

bool readMNIST(MNISTImages &images, bool Test)
{
    string image = "MNIST/train-images.idx3-ubyte";

    if (Test)
    {
        image = "MNIST/t10k-images.idx3-ubyte";
    }

    if (!readImages(image, images))
    {
        cerr << "\nError reading images: " << image << endl;
        return false;
    }

    return true;
}

MNISTImages allocateImages(const MNISTImages &images, int batchSize)
{
    MNISTImages out;
    out.cols = images.cols;
    out.rows = images.rows;
    out.noChannels = images.noChannels;
    out.imageData = nullptr;
    out.noItems = 0;
#ifndef OPENCL
    out.normalizedImageData = new float[images.rows * images.cols * images.noChannels * batchSize];
#else
    out.normalizedImageData = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, images.rows * images.cols * images.noChannels * batchSize * sizeof(float));
#endif
    return out;
}
