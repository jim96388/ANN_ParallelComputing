#pragma once
#include <cstdint>
#include <string>

// This struct stores images from the MINST dataset.
#ifndef OPENCL
struct MNISTImages
{
	int noItems;				// number of images
	int rows, cols;				// number of rows and columns of each image (should be 28*28)
	int noChannels;				// the number of color channels in images
	uint8_t *imageData;			// contains all images data
	float *normalizedImageData; // contains all normalized images data
};
#else
#include "GPU.hpp"
struct MNISTImages
{
	int noItems;					// number of images
	int rows, cols;					// number of rows and columns of each image (should be 28*28)
	int noChannels;					// the number of color channels in images
	cl::Buffer imageData;			// contains all images data
	cl::Buffer normalizedImageData; // contains all normalized images data
};
#endif
// readImages reads the MNIST images and stores them into the images struct from the MNIST dataset specified in the filename
bool readImages(const std::string &filename, MNISTImages &images);

// This function de-allocate memory for images in MNIST dataset.
void destroyMNIST(MNISTImages &);

// This function de-allocate memory for labels in MNIST dataset.
bool readMNIST(MNISTImages &images, bool Test = false);

// This function allocate memory for storing no images
MNISTImages allocateImages(const MNISTImages &images, int no);
