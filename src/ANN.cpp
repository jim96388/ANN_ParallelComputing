#include "../inc/ANN.hpp"
#include <iostream>
#include <cmath>
using namespace std;

// okay, this one is free, I am pretty sure you can do this...
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// TODO: normalize images so each pixel is stored using float (fp32, float32)
// and each image's pixel values are between 0.0f - 1.0f (hint: by dividing each pixel value with 255)
void normalizeImages(MNISTImages &images)
{
    int imagesize = images.noItems * images.rows * images.cols * images.noChannels;
    images.normalizedImageData = new float[imagesize];
    for (int i = 0; i < imagesize; i++)
    {
        images.normalizedImageData[i] = float(images.imageData[i] / 255.0f);
        // cout << images.normalizedImageData[i] << " ";
    }

    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: build your NN model based on the passed in parameters. Hopefully all memory allocation is done here.
// We are going to use only sigmoid activation function, so it is no longer necessary to pass in the function pointers for each layer
Model buildModel(const vector<int> &layerSizes, int batchSize)
{
    Model model;

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
        model.layer[i].output = new float[model.layer[i].output_size * batchSize];
        model.layer[i].weights = new float[model.layer[i].input_size * model.layer[i].output_size];
        model.layer[i].biases = new float[model.layer[i].output_size];
        model.layer[i].output_Err = new float[model.layer[i].output_size * batchSize];
        model.layer[i].theta = new float[model.layer[i].output_size * batchSize];
    }
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
    return model;
}

// TODO: de-allocate all allocated memory (hint: each new must comes with a delete)
void destroyModel(Model &model)
{
    for (int i = 0; i < (model.nolayers - 1); i++)
    {
        delete[] model.layer[i].output;
        delete[] model.layer[i].weights;
        delete[] model.layer[i].biases;

        delete[] model.layer[i].output_Err;
        delete[] model.layer[i].theta;
    }
    delete[] model.layerSizes;
    delete[] model.layer;
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: initialize your model with uniformly distributed random numbers between [-1.0f, 1.0f] for your  weights and biases.
void initializeModel(Model &model)
{
    float max = 1.0f;
    float min = -1.0f;
    srand(time(NULL));
    // srand(1);

    for (int loop = 0; loop < (model.nolayers - 1); loop++)
    {
        for (int i = 0; i < model.layer[loop].input_size * model.layer[loop].output_size; i++)
        {
            model.layer[loop].weights[i] = (max - min) * rand() / (RAND_MAX + 1.0) + min;
        }
        for (int i = 0; i < model.layer[loop].output_size; i++)
        {
            model.layer[loop].biases[i] = (max - min) * rand() / (RAND_MAX + 1.0) + min;
        }
    }
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: Feedforward in batches
void feedForwardBatch(MNISTImages &inputs, int batch, Model &model, MNISTImages &output)
{
    // first time
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < model.layer[0].output_size; i++) // 3
        {
            model.layer[0].output[model.layer[0].output_size * b + i] = 0.0f;
            for (int j = 0; j < model.layer[0].input_size; j++) // 2
            {
                model.layer[0].output[model.layer[0].output_size * b + i] += inputs.normalizedImageData[model.layer[0].input_size * b + j] * model.layer[0].weights[model.layer[0].input_size * i + j];
            }
            model.layer[0].output[model.layer[0].output_size * b + i] += model.layer[0].biases[i];
            model.layer[0].output[model.layer[0].output_size * b + i] = sigmoid(model.layer[0].output[model.layer[0].output_size * b + i]);
            // cout << model.layer[0].output_size * b + i << " ";
            // cout << model.layer[0].output[model.layer[0].output_size * b + i] << " ";
        }
    }

    // loop
    for (int loop = 1; loop < (model.nolayers - 1); loop++)
    {
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < model.layer[loop].output_size; i++) // 2
            {
                model.layer[loop].output[model.layer[loop].output_size * b + i] = 0.0f;
                for (int j = 0; j < model.layer[loop].input_size; j++) // 3
                {
                    // cout << " \n 輸入 : " << model.layer[loop - 1].output[model.layer[loop - 1].output_size * b + i] << " " << model.layer[loop].weights[model.layer[loop].input_size * i + j];
                    model.layer[loop].output[model.layer[loop].output_size * b + i] += model.layer[loop - 1].output[model.layer[loop - 1].output_size * b + j] * model.layer[loop].weights[model.layer[loop].input_size * i + j]; // 3=model.layer[loop].input_size
                    // cout << "\n 計算 : " << model.layer[loop].output[model.layer[loop].output_size * b + i];
                }
                model.layer[loop].output[model.layer[loop].output_size * b + i] += model.layer[loop].biases[i];
                // cout << "\n\n 結果 : " << model.layer[loop].output[model.layer[loop].output_size * b + i];
                model.layer[loop].output[model.layer[loop].output_size * b + i] = sigmoid(model.layer[loop].output[model.layer[loop].output_size * b + i]);
                // cout << model.layer[loop].output[model.layer[loop].output_size * b + i] << " ";
            }
        }
    }
    for (int i = 0; i < model.layer[model.nolayers - 2].output_size * batch; i++)
    {
        output.normalizedImageData[i] = model.layer[model.nolayers - 2].output[i];
    }
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
}

// TODO: Back-propagate in batches. Note the function returns MAE (mean absolute error) rather than SSE (sum squared error)
float backPropagateBatch(MNISTImages &image, int no, int batch, Model &model, MNISTImages &inputs, float alpha)
{
    float MAE = 0.0f;
    auto imagesize = image.cols * image.rows * image.noChannels;
    for (int b = 0; b < batch; b++) //? 2
    {
        for (int i = 0; i < model.layer[model.nolayers - 2].output_size; i++) //? 2
        {
            // cout << model.layer[model.nolayers - 2].output[i] << " " << i << " ";
            model.layer[model.nolayers - 2].output_Err[model.layer[model.nolayers - 2].output_size * b + i] =
                model.layer[model.nolayers - 2].output[model.layer[model.nolayers - 2].output_size * b + i] - image.normalizedImageData[no * imagesize + model.layer[model.nolayers - 2].output_size * b + i];
            // cout << model.layer[model.nolayers - 2].output_Err[i] << " ";
        }
    }

    for (int b = 0; b < batch; b++) //? 2
    {
        for (int i = 0; i < model.layer[model.nolayers - 2].output_size; i++) //? 2
        {
            MAE += fabs(model.layer[model.nolayers - 2].output_Err[model.layer[model.nolayers - 2].output_size * b + i]);
        }
    }
    MAE /= imagesize * batch;

    for (int loop = model.nolayers - 2; loop > 0; loop--) // 1
    {
        // #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < model.layer[loop].output_size; j++) // 2
            {
                // #pragma omp parallel for
                for (int i = 0; i < model.layer[loop].input_size; i++) // 3
                {
                    model.layer[loop - 1].output_Err[model.layer[loop].input_size * b + i] = 0.0f;
                    //? theta = (output*(1-output))*誤差
                    model.layer[loop].theta[model.layer[loop].output_size * b + j] = (model.layer[loop].output[model.layer[loop].output_size * b + j] * (1 - model.layer[loop].output[model.layer[loop].output_size * b + j])) * model.layer[loop].output_Err[model.layer[loop].output_size * b + j];
                    //? 傳遞物差: 下一層誤差= theta * 原權重
                    model.layer[loop - 1].output_Err[model.layer[loop].input_size * b + i] += model.layer[loop].theta[model.layer[loop].output_size * b + j] * model.layer[loop].weights[j * model.layer[loop].input_size + i];
                    //? 權重=權重-學習率*(theta*output)
                    // cout << model.layer[loop].theta[model.layer[loop].output_size * b + j] << " ";
                    // cout << model.layer[loop].weights[j * model.layer[loop].input_size + i] << " ";
                    model.layer[loop].weights[j * model.layer[loop].input_size + i] = model.layer[loop].weights[j * model.layer[loop].input_size + i] - alpha * (model.layer[loop].theta[model.layer[loop].output_size * b + j] * model.layer[loop - 1].output[model.layer[loop].input_size * b + i] / batch);
                }
                //? 偏差=偏差-學習率*(theta)
                model.layer[loop].biases[j] = model.layer[loop].biases[j] - alpha * (model.layer[loop].theta[model.layer[loop].output_size * b + j] / batch);
            }
        }
    }

    for (int b = 0; b < batch; b++) //? 批次
    {
        for (int o = 0; o < model.layer[0].output_size; o++) // 3
        {
            model.layer[0].theta[model.layer[0].output_size * b + o] = (model.layer[0].output[model.layer[0].output_size * b + o] * (1 - model.layer[0].output[model.layer[0].output_size * b + o])) * model.layer[0].output_Err[model.layer[0].output_size * b + o];
            for (int i = 0; i < model.layer[0].input_size; i++) // 2
            {
                model.layer[0].weights[model.layer[0].input_size * o + i] = model.layer[0].weights[model.layer[0].input_size * o + i] - alpha * (model.layer[0].theta[model.layer[0].output_size * b + o] * inputs.normalizedImageData[model.layer[0].input_size * b + i] / batch);
            }
            model.layer[0].biases[o] = model.layer[0].biases[o] - alpha * (model.layer[0].theta[model.layer[0].output_size * b + o] / batch);
        }
    }
    // cout << "\nPlease implement '" << __func__ << "' function. Line " << __LINE__ << "@ " << __FILE__;
    return MAE;
}
