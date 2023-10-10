#include "../inc/GPU.hpp"
#include <string>
#include <vector>
#include <iostream>
// Define your GPU kernels here
auto src = R"CLC(
    
    float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x)); 
    }

    __kernel
    void myAtomicAddG(__global float *addr, float val) {
        union {
            uint u32;
            float f32;
        } current, expected, next;
        do {
            current.f32 =*addr;
            next.f32 = current.f32 + val;
            expected.u32 = current.u32;
            current.u32 = atomic_cmpxchg( (__global uint*) addr, expected.u32, next.u32 );
        } while( current.u32 != expected.u32 );
    }

    __kernel
    void k_normalizeImages(uint n, __global uchar* images_imageData, __global float* images_normalizedImageData){
        int g = get_global_id(0);
        if( g < n ){
            images_normalizedImageData[g] = (float)(images_imageData[g]/255.0f);
            // printf(" %f ",images_normalizedImageData[g] );
        }
    }
    
    __kernel
    void k_addNoise(uint imagesize_b, uint offset, uint imagesize, __global float* allSourceImages, __global float* output, __global float* noise_buf){
        uint g = get_global_id(0);
        if( g < imagesize_b ){
            output[g] = allSourceImages[offset * imagesize + g] + noise_buf[g];
            if(output[g] > 1.0f){
                output[g] -= 1.0f;
            }
            // printf("%g ",output[g]);
        }
    }

    __kernel
    void k_feedForward_first(uint batch,  uint input_size, uint output_size, __global float* output, __global float* inputs_normalizedImageData, __global float* weights, __global float* biases ){
        auto o = get_global_id(0); 
        auto b = get_global_id(1);   
        if( b < batch ){ 
            if( o < output_size ){ 
                output[output_size * b + o] = 0.0f;
                for(int i = 0; i < input_size ; i++ ){ 
                    output[output_size * b + o] += inputs_normalizedImageData[input_size * b + i] * weights[input_size * o + i]; 
                    //  printf("%g ",weights[input_size * o + i]);
                }
                output[output_size * b + o] += biases[o];    
                // printf("%g ",biases[o]);
                output[output_size * b + o] = sigmoid(output[output_size * b + o]);
            }
        }
    }

    __kernel
    void k_feedForward_loop(  uint batch,  uint input_size, uint output_size, __global float* output, __global float* output_lastloop, __global float* weights, __global float* biases ){
        auto o = (uint)get_global_id(0); 
        auto b = (uint)get_global_id(1); 
        if( b < batch ){
            if( o < output_size ){
                output[output_size * b + o] = 0.0f;
                for(int i = 0; i < input_size ; i++ ){
                    output[output_size * b + o] += output_lastloop[input_size * b + i] * weights[input_size * o + i]; 
                }
                output[output_size * b + o] += biases[o];
                output[output_size * b + o] = sigmoid(output[output_size * b + o]);
            }
        }
    }

    __kernel
    void k_feedForward_return(uint last_outputsize, __global float* output, __global float* output_normalizedImageData){
        auto i = get_global_id(0);
        if(i < last_outputsize){
            output_normalizedImageData[i]=output[i];
            // printf("%g",output_return[i]);
        }
    }

    __kernel
    void k_calculateMAE( uint no ,uint batch, uint imagesize, uint input_size, uint output_size, __global float* output, __global float* output_Err, __global float* image,__global float* atomic_temp ){
        auto o = (uint)get_global_id(0); 
        auto b = (uint)get_global_id(1); 
        if( b < batch ){ 
            if( o < output_size ){ 
                output_Err[output_size * b + o] = output[output_size * b + o] - image[ imagesize* no + o]; 
                // printf(" %g ",image[ imagesize* no + o]);
                if(output_Err[output_size * b + o] >= 0.0f){
                    myAtomicAddG( &atomic_temp[0] ,output_Err[output_size * b + o] /batch);
                }
                else{
                    myAtomicAddG( &atomic_temp[0] ,-output_Err[output_size * b + o] /batch);
                }
                // printf(" %g ",atomic_temp[0]);
            }
        }   
    }

    __kernel
    void k_backPropagate_loop( uint offset, uint batch, float alpha, uint input_size, uint output_size, __global float* output, __global float* output_nextloop, __global float* weights, __global float* biases
    , __global float* output_Err, __global float* output_Err_nextloop, __global float* theta, __global float* weights_temp , __global float* biases_temp){
        auto i = (uint)get_global_id(0); 
        auto b = (uint)get_global_id(1); 
        if( b < batch ){ 
            if( i < input_size ){ 
                output_Err_nextloop[input_size * b + i] = 0.0f;
                for (int o = 0; o < output_size; o++) {
                    theta[output_size * b + o] =(output[output_size * b + o] * (1 - output[output_size * b + o])) * output_Err[output_size * b + o];//! ok
                    // printf("%g \n", theta[output_size * b + o]);
                    output_Err_nextloop[input_size * b + i] += theta[output_size * b + o] * weights[input_size * o + i];
                    // printf("%g ",output_Err_nextloop[input_size * b + i] );
                }
                for (int o = 0; o < output_size; o++)
                {
                    // printf("%g %g  \n",weights[o * input_size + i] , theta[output_size * b + o] * output_nextloop[input_size * b + i]);
                    weights_temp[o * input_size + i] = weights[o * input_size + i];
                    myAtomicAddG( &weights_temp[o * input_size + i], -alpha * (theta[output_size * b + o] * output_nextloop[input_size * b + i] / batch ));
                    weights[o * input_size + i] = weights_temp[o * input_size + i];
                    // printf("%g \n",weights[o * input_size + i]);
                }
            }
        }
    }
    
        __kernel
    void k_backPropagate_loop2( uint offset, uint batch, float alpha, uint input_size, uint output_size, __global float* output, __global float* output_nextloop, __global float* weights, __global float* biases
    , __global float* output_Err, __global float* output_Err_nextloop, __global float* theta, __global float* weights_temp , __global float* biases_temp){
        auto o = (uint)get_global_id(0); 
        auto b = (uint)get_global_id(1); 
        if( b < batch ){
            if( o < output_size ){
                // biases[o] -= alpha * (theta[output_size * b + o] / batch);
                biases_temp[o]= biases[o];
                myAtomicAddG( &biases_temp[o],-alpha* (theta[output_size * b + o]/ batch));
                biases[o]= biases_temp[o];
                // printf("%g \n",biases[i]);
            }    
        }
    }

    __kernel
    void k_backPropagate_first( uint offset, uint batch, float alpha,  uint input_size, uint output_size, __global float* output, __global float* weights, __global float* biases, 
    __global float* output_Err, __global float* theta, __global float* input, __global float* weights_temp , __global float* biases_temp){
        auto o = (uint)get_global_id(0); 
        auto b = (uint)get_global_id(1); 
        if( b < batch ){
            if( o < output_size ){
                theta[output_size * b + o] = (output[output_size * b + o] * (1 - output[output_size * b + o])) * output_Err[output_size * b + o];
                // printf(" %g \n",theta[output_size * b + o] );  
                for (int i = 0; i < input_size; i++) // 2
                {
                    // weights[input_size * o + i] = weights[input_size * o + i] - alpha * (theta[output_size * b + o] * input[b*input_size+i] / batch);
                    // weights[input_size * o + i] = weights[input_size * o + i] - alpha * (theta[output_size * b + o] * inputs[b][i] / batch);
                    // printf(" %g %g \n " ,theta[output_size * b + o],input[input_size* b + i]);
                    
                    weights_temp[o * input_size + i] = weights[o * input_size + i];
                    myAtomicAddG( &weights_temp[o * input_size + i], - alpha * (theta[output_size * b + o] * input[input_size * b + i] / batch ));
                    weights[o * input_size + i] = weights_temp[o * input_size + i];
                    // printf("%g \n",weights[o * input_size + i]);
                    
                }
                // biases[o] = biases[o] - alpha * (theta[output_size * b + o] / batch);
                biases_temp[o]= biases[o];
                myAtomicAddG( &biases_temp[o],-alpha* (theta[output_size * b + o]/ batch));
                biases[o]= biases_temp[o];
                // printf("%g \n",biases[o]);
            }
        }
    }
)CLC";

// Utility function to get a device
auto getDevice(const char *vendor, size_t MB)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto &p : platforms)
    {
        if (p.getInfo<CL_PLATFORM_VENDOR>().find(vendor) == std::string::npos)
            continue;
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        for (auto d : devices)
        {
            if (d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() < MB * 1024ul * 1024ul)
                continue;
            return d;
        }
    }
    throw cl::Error(CL_INVALID_DEVICE, "No such vendor or no device has needed global memory size");
}

extern GPU gpu;
void peek(std::string label, cl::Buffer &vector, cl::size_type sz)
{
    std::cout << label << ":[";
    auto tmp = (float *)gpu.cmdQueue.enqueueMapBuffer(vector, CL_TRUE, CL_MAP_READ, 0, sz * sizeof(float));
    for (auto i = 0ul; i < sz; ++i)
        std::cout << tmp[i] << " ";
    std::cout << "]\n";
    gpu.cmdQueue.enqueueUnmapMemObject(vector, tmp);
}

GPU initGPU(int wgsize)
{
    GPU gpu;
    // auto device = getDevice("Intel", 256);
    auto device = getDevice("NVIDIA", 256);
    // auto device = getDevice("Mesa", 256);
    auto ctx = cl::Context(device);
    auto cmdQueue = cl::CommandQueue(ctx, device);

    auto prog = cl::Program(ctx, src);
    try
    {
        prog.build();
    }
    catch (cl::Error &)
    {
        std::cerr << "\n"
                  << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        throw cl::Error(CL_INVALID_PROGRAM, "Failed to build kernel");
    }

    // Store needed stuff in GPU struct
    gpu.device = device;
    gpu.ctx = ctx;
    gpu.cmdQueue = cmdQueue;
    gpu.workGroupSize = wgsize;
    gpu.prg = prog;

    return gpu;
}
