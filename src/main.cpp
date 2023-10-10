#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <random>
using namespace std;
#include "../inc/ANN.hpp"
#include "../inc/CIFAR10.hpp"
#include "../inc/MNIST.hpp"

#ifdef OPENCL
#include "../inc/GPU.hpp"
GPU gpu;

#else
#ifdef OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>

cv::Mat prepareMat(int rows, int cols, int type, float *image)
{
    if (type == CV_32FC1)
    {
        return cv::Mat(rows, cols, type, image);
    }
    else
    {
        vector<float> convertedImage;
        auto size = rows * cols;
        convertedImage.reserve(size * 3);
        for (auto i = 0; i < size; ++i)
        {
            for (auto c = 0; c < 3; ++c)
                convertedImage.push_back(image[c * size + i]);
        }
        auto img = cv::Mat(rows, cols, type, convertedImage.data());
        return cv::Mat(rows, cols, type, convertedImage.data());
    }
}
#endif
#endif

// readData declares a function pointer type, so that we can use function pointer to read either MNIST or CIFAR10 dataset.
using readData = bool (*)(MNISTImages &images, bool Test);

// Let's make noisy images
void addNoise(const MNISTImages &allSourceImages, int offset, int b, MNISTImages &output, float probability = 0.2f)
{
#ifndef OPENCL

    float max = 1.0f;
    float min = 0.0f;
    srand(time(NULL));
    // srand(1);
    // (max - min) * rand() / (RAND_MAX + 1.0) + min;
    auto imagesize = allSourceImages.cols * allSourceImages.rows * allSourceImages.noChannels;
    for (auto i = 0; i < imagesize * b; ++i)
    {
        output.normalizedImageData[i] = allSourceImages.normalizedImageData[offset * imagesize + i];
        float dice = (max - min) * rand() / (RAND_MAX + 1.0) + min;
        if (dice <= probability)
        {
            float noise = (max - min) * rand() / (RAND_MAX + 1.0f) + min;
            output.normalizedImageData[i] += noise;
            if (output.normalizedImageData[i] > 1.0f)
            {
                output.normalizedImageData[i] -= 1.0f;
            }
        }
    }

#else

    auto imagesize = allSourceImages.cols * allSourceImages.rows * allSourceImages.noChannels;
    auto imagesize_b = imagesize * b;
    default_random_engine eng(1);
    uniform_real_distribution<float> distr(0.0f, 1.0f);
    vector<float> noise;
    for (int i = 0; i < imagesize_b; i++)
    {
        float dice = distr(eng);
        if (dice < probability)
        {
            float X = distr(eng);
            noise.push_back(X);
        }
        else
        {
            noise.push_back(0);
        }
        // cout << noise[i] << " ";
    }
    auto noise_buf = cl::Buffer(gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * imagesize_b, noise.data());
    cl::KernelFunctor<cl_uint, cl_uint, cl_uint, const cl::Buffer &, cl::Buffer &, cl::Buffer &> k_addNoise(gpu.prg, "k_addNoise");
    auto gobal = cl::NDRange((imagesize_b + 255) / 256 * 256);
    auto workgroup = cl::NDRange(256);
    auto config = cl::EnqueueArgs(gpu.cmdQueue, gobal, workgroup);
    k_addNoise(config, imagesize_b, offset, imagesize, allSourceImages.normalizedImageData, output.normalizedImageData, noise_buf);
    noise_buf = cl::Buffer();

#endif
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
    // This function will takes b unaltered images from the allSourceImages starting from offset
    // the percentage specifies the probability on each pixel to add noises
    // the noise is added by adding a random number in the range of [0, 1] to the original pixel value
    // Note that after adding the noise, the pixel value must be capped at 1.0 (no larger than 1.0)
}

vector<int> setupArchitecture(int which, int inputSize)
{
    switch (which)
    {
    case 0:
        return {inputSize, 96, inputSize};
    case 1:
        return {inputSize, 192, 48, 192, inputSize};
    case 2:
        return {inputSize, 768, 192, 48, 192, 768, inputSize};
    case 3:
        return {inputSize, 1536, 768, 384, 192, 96, 48, 96, 192, 386, 768, 1536, inputSize};
    default:
        return {inputSize, 192, inputSize};
    }
}

void saveSamples(readData reader, Model &autoencoder, int epoch)
{
// #if defined(OPENCV) && !defined(OPENCL)
#if defined(OPENCV)
    MNISTImages images;
    if (!reader(images, true))
        return;
    int type = CV_32FC1;
    if (images.noChannels == 3)
        type = CV_32FC3;

    normalizeImages(images);
    auto noisyImages = allocateImages(images, 1);
    auto reconstructedImages = allocateImages(images, 1);

    auto no = 0;
    const auto sz = images.rows * images.cols * images.noChannels;
    cv::Mat big;
    for (auto y = 0; y < 20; ++y)
    {
        auto rowImage = cv::Mat();
        for (auto x = 0; x < 10; ++x, ++no)
        {
            auto tmp = cv::Mat();
            addNoise(images, no, 1, noisyImages);
            feedForwardBatch(noisyImages, 1, autoencoder, reconstructedImages);
            auto img1 = prepareMat(images.rows, images.cols, type, images.normalizedImageData + no * sz).clone();
            auto img2 = prepareMat(images.rows, images.cols, type, noisyImages.normalizedImageData).clone();
            auto img3 = prepareMat(images.rows, images.cols, type, reconstructedImages.normalizedImageData).clone();

            cv::hconcat(img1, img2, tmp);
            cv::hconcat(tmp, img3, tmp);
            if (rowImage.dims == 0)
                rowImage = tmp;
            else
                cv::hconcat(rowImage, tmp, rowImage);
        }
        big.push_back(rowImage);
    }
    cv::imwrite("samples-"s + to_string(epoch) + ".tiff", big);
#endif
}

int main(int argc, const char **argv)
{
    // auto readDataSet = readCIFAR10;
    auto readDataSet = readMNIST;
    auto batchSize = 1;
    auto noEpoches = 5;
    auto learningRate = 0.1f;
    auto whichArchitecture = 0;
    if (argc > 1)
        whichArchitecture = atoi(argv[1]);
    if (argc > 2)
        batchSize = atoi(argv[2]);
    if (argc > 3)
        noEpoches = atoi(argv[3]);
    if (argc > 4)
        learningRate = (float)atof(argv[4]);
    if (argc > 5)
        readDataSet = readCIFAR10;

#ifdef OPENCL
    gpu = initGPU(256);
#endif
    // double T_total = omp_get_wtime();
    // double T = 0, T1 = 0, T2 = 0, T3 = 0, T4 = 0;
    // T = omp_get_wtime();
    // Get Training Data
    MNISTImages images;
    if (!readDataSet(images, false))
        return 255;
    normalizeImages(images);
    auto noisyImages = allocateImages(images, batchSize);
    auto reconstructedImages = allocateImages(images, batchSize);

    // Prepare Autoencoder Model
    auto sizes = setupArchitecture(whichArchitecture, images.cols * images.rows * images.noChannels);
    auto autoencoder = buildModel(sizes, batchSize);
    initializeModel(autoencoder);
    // T1 += omp_get_wtime() - T;
    // Training
    for (auto epoch = 1; epoch <= noEpoches; ++epoch)
    {
        auto R = 0.0f;
        for (auto no = 0; no < images.noItems; no += batchSize)
        {
            auto batch = std::min(batchSize, images.noItems - no);
            // T = omp_get_wtime();
            // Let's make dirty images
            addNoise(images, no, batch, noisyImages);
            // T2 += omp_get_wtime() - T;

            // T = omp_get_wtime();
            // Feed the dirty image to the network
            feedForwardBatch(noisyImages, batch, autoencoder, reconstructedImages);
            //* noisyImages => input  reconstructedImages => output
            // T3 += omp_get_wtime() - T;

            // T = omp_get_wtime();
            // The target is the original image (before adding noise).  Let's update the weights and biases
            auto r = backPropagateBatch(images, no, batch, autoencoder, noisyImages, learningRate);
            // T4 += omp_get_wtime() - T;

            R += r;
            if (no % 250 == 0)
            {
                cout << char(13) << no << "/" << images.noItems << ":" << r * 100.0f << "%     " << flush;
            }
        }
        // cout << "\n T1:"
        //  << T1 << " T2: " << T2 << " T3: " << T3 << " T4: " << T4 << endl;
        cout << char(13) << epoch << ":" << R * 100.0f / images.noItems << "%                " << endl;
        saveSamples(readDataSet, autoencoder, epoch);
        // each epoch uses decreasing learingRate to fine tune parameters
        learningRate *= 0.5f;
    }

    destroyMNIST(images);
    destroyMNIST(noisyImages);
    destroyMNIST(reconstructedImages);
    destroyModel(autoencoder);

    // cout << "\n Total time: " << (omp_get_wtime() - T_total);
    return 0;
}
