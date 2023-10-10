#include "../inc/ANN.hpp"
#include "../inc/GPU.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cmath>
#include <random>
using namespace std;

extern GPU gpu;

// okay, this one is free, I am pretty sure you can do this...
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// TODO: normalize images so each pixel is stored using float (fp32, float32)
// and each image's pixel values are between 0.0f - 1.0f (hint: by dividing each pixel value with 255)
void normalizeImages(MNISTImages &images)
{
#ifndef OPENCL
#else
    //* 單迴圈版
    auto n = images.noItems * images.cols * images.rows * images.noChannels;
    // images.normalizedImageData = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * n);
    auto gobal = cl::NDRange((n + 255) / 256 * 256);
    auto workgroup = cl::NDRange(256);
    auto k_normalizeImages = cl::KernelFunctor<cl_uint, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_normalizeImages");
    auto config = cl::EnqueueArgs(gpu.cmdQueue, gobal, workgroup);
    k_normalizeImages(config, n, images.imageData, images.normalizedImageData);
#endif
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: build your NN model based on the passed in parameters. Hopefully all memory allocation is done here.
// We are going to use only sigmoid activation function, so it is no longer necessary to pass in the function pointers for each layer
Model buildModel(const vector<int> &layerSizes, int batchSize)
{
    Model model;
#ifndef OPENCL
#else
    model.nolayers = layerSizes.size();
    model.layerSizes = new int[layerSizes.size()];
    for (ulong i = 0; i < layerSizes.size(); i++)
    {
        model.layerSizes[i] = layerSizes[i]; //{ 2 3 2 }
    }
    model.layer = new Layer[model.nolayers - 1];
    for (int i = 0; i < (model.nolayers - 1); i++)
    {
        model.layer[i].input_size = model.layerSizes[i];
        model.layer[i].output_size = model.layerSizes[i + 1];

        //? 以下僅開空間 無資料
        model.layer[i].output = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * model.layer[i].output_size * batchSize);
        model.layer[i].output_Err = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * model.layer[i].output_size * batchSize);
        model.layer[i].theta = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * model.layer[i].output_size * batchSize);

        model.layer[i].weights = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * model.layer[i].input_size * model.layer[i].output_size);
        model.layer[i].biases = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * model.layer[i].output_size);
    }
#endif

    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
    return model;
}

// TODO: de-allocate all allocated memory (hint: each new must comes with a delete)
void destroyModel(Model &model)
{
#ifndef OPENCL
#else
    for (int i = 0; i < (model.nolayers - 1); i++)
    {
        model.layer[i].output = cl::Buffer();
        model.layer[i].output_Err = cl::Buffer();
        model.layer[i].theta = cl::Buffer();

        model.layer[i].weights = cl::Buffer();
        model.layer[i].biases = cl::Buffer();
    }
    delete[] model.layerSizes;
    delete[] model.layer;
#endif

    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: initialize your model with uniformly distributed random numbers between [-1.0f, 1.0f] for your  weights and biases.
void initializeModel(Model &model)
{
#ifndef OPENCL
#else
    // float max = 1.0f;
    // float min = -1.0f;
    // srand(time(NULL));
    // srand(1);
    // (max - min) * rand() / (RAND_MAX + 1.0) + min;
    default_random_engine eng(1);
    uniform_real_distribution<float> distr(1.0f, -1.0f);
    for (int loop = 0; loop < (model.nolayers - 1); loop++)
    {
        // 準備隨機參數空間
        float *weights_data = new float[model.layer[loop].input_size * model.layer[loop].output_size];
        float *biases_data = new float[model.layer[loop].output_size];
        for (int i = 0; i < model.layer[loop].input_size * model.layer[loop].output_size; i++)
        {
            weights_data[i] = distr(eng);
        }
        for (int i = 0; i < model.layer[loop].output_size; i++)
        {
            biases_data[i] = distr(eng);
        }

        gpu.cmdQueue.enqueueWriteBuffer(model.layer[loop].weights, CL_TRUE, 0, model.layer[loop].input_size * model.layer[loop].output_size * sizeof(float), weights_data);
        gpu.cmdQueue.enqueueWriteBuffer(model.layer[loop].biases, CL_TRUE, 0, model.layer[loop].output_size * sizeof(float), biases_data);

        delete[] weights_data;
        delete[] biases_data;
    }
#endif

    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: Feedforward in batches
void feedForwardBatch(MNISTImages &inputs, int batch, Model &model, MNISTImages &output)
{
#ifndef OPENCL
#else
    //! k_feedForward_first
    auto global_first = cl::NDRange(((model.layer[0].output_size + 15) / 16 * 16), ((batch + 15) / 16 * 16));
    auto workgroup_first = cl::NDRange(16, 16);
    auto k_feedForward_first = cl::KernelFunctor<cl_uint, cl_uint, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_feedForward_first");
    auto config_first = cl::EnqueueArgs(gpu.cmdQueue, global_first, workgroup_first);
    k_feedForward_first(config_first, batch, model.layer[0].input_size, model.layer[0].output_size, model.layer[0].output, inputs.normalizedImageData, model.layer[0].weights, model.layer[0].biases);

    for (int loop = 1; loop < (model.nolayers - 1); loop++)
    {
        //! k_feedForward_loop
        auto global_loop = cl::NDRange(((model.layer[loop].output_size + 15) / 16 * 16), ((batch + 15) / 16 * 16));
        auto workgroup_loop = cl::NDRange(16, 16);
        auto k_feedForward_loop = cl::KernelFunctor<cl_uint, cl_uint, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_feedForward_loop");
        auto config_loop = cl::EnqueueArgs(gpu.cmdQueue, global_loop, workgroup_loop);
        k_feedForward_loop(config_loop, batch, model.layer[loop].input_size, model.layer[loop].output_size, model.layer[loop].output, model.layer[loop - 1].output, model.layer[loop].weights, model.layer[loop].biases);
    }

    auto global_return = cl::NDRange((model.layer[model.nolayers - 2].output_size * batch + 255) / 256 * 256);
    auto workgroup_return = cl::NDRange(256);
    auto k_feedForward_return = cl::KernelFunctor<cl_uint, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_feedForward_return");
    auto config_return = cl::EnqueueArgs(gpu.cmdQueue, global_return, workgroup_return);
    k_feedForward_return(config_return, model.layer[model.nolayers - 2].output_size * batch, model.layer[model.nolayers - 2].output, output.normalizedImageData);
#endif

    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: Back-propagate in batches. Note the function returns MAE (mean absolute error) rather than SSE (sum squared error)
float backPropagateBatch(MNISTImages &image, int no, int batch, Model &model, MNISTImages &inputs, float alpha)
{
    float MAE = 0.0f;
#ifndef OPENCL
#else
    //! MAE計算部分
    auto atomic_temp = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float));
    auto imagesize = image.cols * image.rows * image.noChannels;
    int last = model.nolayers - 2;

    auto global_SSE = cl::NDRange(((model.layer[last].output_size + 15) / 16 * 16), ((batch + 15) / 16 * 16));
    auto workgroup_SSE = cl::NDRange(16, 16);
    auto k_calculateMAE = cl::KernelFunctor<cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_calculateMAE");
    auto config_SSE = cl::EnqueueArgs(gpu.cmdQueue, global_SSE, workgroup_SSE);
    k_calculateMAE(config_SSE, no, batch, imagesize, model.layer[last].input_size, model.layer[last].output_size,
                   model.layer[last].output, model.layer[last].output_Err, image.normalizedImageData, atomic_temp);

    gpu.cmdQueue.enqueueReadBuffer(atomic_temp, CL_TRUE, 0, sizeof(MAE), &MAE);
    MAE /= imagesize * batch;
    atomic_temp = cl::Buffer();

    //! 向後loop層
    for (int loop = model.nolayers - 2; loop > 0; loop--)
    {
        auto weights_temp = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * (model.layer[loop].input_size * model.layer[loop].output_size));
        auto biases_temp = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * (model.layer[loop].output_size));
        auto global_loop1 = cl::NDRange(((model.layer[loop].input_size + 15) / 16 * 16), ((batch + 15) / 16 * 16));
        auto workgroup_loop1 = cl::NDRange(16, 16);
        auto config_loop1 = cl::EnqueueArgs(gpu.cmdQueue, global_loop1, workgroup_loop1);

        auto k_backPropagate_loop = cl::KernelFunctor<cl_uint, cl_uint, cl_float, cl_uint, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_backPropagate_loop");
        k_backPropagate_loop(config_loop1, no, batch, alpha, model.layer[loop].input_size, model.layer[loop].output_size, model.layer[loop].output, model.layer[loop - 1].output, model.layer[loop].weights,
                             model.layer[loop].biases, model.layer[loop].output_Err, model.layer[loop - 1].output_Err, model.layer[loop].theta, weights_temp, biases_temp);

        auto global_loop2 = cl::NDRange(((model.layer[loop].output_size + 15) / 16 * 16), ((batch + 15) / 16 * 16));
        auto workgroup_loop2 = cl::NDRange(16, 16);
        auto config_loop2 = cl::EnqueueArgs(gpu.cmdQueue, global_loop2, workgroup_loop2);

        auto k_backPropagate_loop2 = cl::KernelFunctor<cl_uint, cl_uint, cl_float, cl_uint, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_backPropagate_loop2");
        k_backPropagate_loop2(config_loop1, no, batch, alpha, model.layer[loop].input_size, model.layer[loop].output_size, model.layer[loop].output, model.layer[loop - 1].output, model.layer[loop].weights,
                              model.layer[loop].biases, model.layer[loop].output_Err, model.layer[loop - 1].output_Err, model.layer[loop].theta, weights_temp, biases_temp);
        weights_temp = cl::Buffer();
        biases_temp = cl::Buffer();
    }

    auto weights_temp = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * (model.layer[0].input_size * model.layer[0].output_size));
    auto biases_temp = cl::Buffer(gpu.ctx, CL_MEM_READ_WRITE, sizeof(float) * (model.layer[0].output_size));
    auto global_first = cl::NDRange(((model.layer[0].output_size + 15) / 16 * 16), ((batch + 15) / 16 * 16));
    auto workgroup_first = cl::NDRange(16, 16);
    auto config_first = cl::EnqueueArgs(gpu.cmdQueue, global_first, workgroup_first);
    auto k_backPropagate_first = cl::KernelFunctor<cl_uint, cl_uint, cl_float, cl_uint, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(gpu.prg, "k_backPropagate_first");
    k_backPropagate_first(config_first, no, batch, alpha, model.layer[0].input_size, model.layer[0].output_size, model.layer[0].output, model.layer[0].weights, model.layer[0].biases,
                          model.layer[0].output_Err, model.layer[0].theta, inputs.normalizedImageData, weights_temp, biases_temp);
    weights_temp = cl::Buffer();
    biases_temp = cl::Buffer();

// cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
#endif
    return MAE;
}
