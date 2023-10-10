#pragma once
#include "MNIST.hpp"
#include <vector>
void normalizeImages(MNISTImages &images);

// I created a Layer struct to store layer-related information
// (input size, output size, weights, biases, output, and activation function.)
// It should be noted that you should allow each layer to use a different activation function.
#ifndef OPENCL
struct Layer
{
    int input_size;
    int output_size;

    float *weights;    // 權重(一維)
    float *biases;     // 常數
    float *output;     // 輸出
    float *output_Err; // 輸出之誤差
    float *theta;
};
#else
struct Layer
{
    int input_size;
    int output_size;

    cl::Buffer weights;
    cl::Buffer biases;
    cl::Buffer output;
    cl::Buffer output_Err;
    cl::Buffer theta;
};
#endif
// I created this struct Model to store model information (e.g. number of layers, details about each layer)
struct Model
{
    Layer *layer;
    int nolayers;    //? 神經元層數
    int *layerSizes; //? 神經元節點數
};

// buildModels allocates memory and create all layers of NN based on given parameters:
//      noLayers: number of layers (of neurons)
//      layerSizes: a 1-D array contains the number of neurons at each layer.
Model buildModel(const std::vector<int> &sizes, int bathSize = 1);

// destroyModel de-allocates allocated memory in buildModel.
void destroyModel(Model &);

// initializeModel initializes all weights in all layers.
// The weights are uniformly distributed random numbers in [-1, 1].
void initializeModel(Model &model);

// feedForward carries out the feed-forward process with miniBachSize inputs
void feedForwardBatch(MNISTImages &input, int batch, Model &model, MNISTImages &output);

// backward propagate
float backPropagateBatch(MNISTImages &target, int no, int batch, Model &model, MNISTImages &inputs, float alpha);