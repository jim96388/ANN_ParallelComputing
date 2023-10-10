#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

struct GPU {
	cl::Device device;
	cl::CommandQueue cmdQueue;
	cl::Context ctx;
	cl::Program prg; 

	int workGroupSize; 
};

GPU initGPU(int workGroupSize); 
