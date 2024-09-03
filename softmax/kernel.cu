#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cmath>
#include "cuda_runtime.h"

#define CHECK(func) do \
{ \
	cudaError_t err = func \
	if (err != cudaSuccess) \
	{   \
		printf("error: %s, %d \n", cudaGetErrorString(err), __LINE__); \
	} while(0) \
}

#define BLOCK_SIZE 256

#define M 1024
#define N 2048


/*
Softmax:
	
	主要是两步 reduce:

	1. 求每行最大值 max_val:

	2. 求每行指数和：
	
		sum = sum(expf(xi - max_val))

	3. 求值：
		
		xi = exp(xi-max_val) / sum
*/

/*
	input, output shape: (M, N)
*/

void cpu_softmax(float* input, float* output, int m, int n)
{
	
	for (int i = 0; i < m; ++i)
	{
		// max_val:
		float max_val = 0.0f;

		for (int j = 0; j < n; ++j)
		{
			max_val = fmaxf(input[i * n + j], max_val);
		}

		// sum:
		float sum = 0.0f;
		for (int j = 0; j < n; ++j)
		{
			sum += expf(input[i * n + j] - max_val);
		}

		// value:
		for (int j = 0; j < n; ++j)
		{
			output[i*n+j] = expf(input[i * n + j] - max_val) / sum;
		}
	}

}


// kernel0: 每行并行计算， 一个线程处理一行数据：
__global__ void softmax_thread(float* input, float* output, int m, int n)
{
	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < m)
	{
		float* inp = input + row * n;

		// max_val:
		float max_val = 0.0f;

		for (int j = 0; j < n; ++j)
		{
			max_val = fmaxf(inp[j], max_val);
		}

		// sum:
		float sum = 0.0f;
		for (int j = 0; j < n; ++j)
		{
			sum += expf(inp[j] - max_val);
		}

		// value:
		for (int j = 0; j < n; ++j)
		{
			output[row*n + j] = expf(inp[j] - max_val) / sum;
		}
	}

}


// kernel1: 
// 上述对input的读取都是在 global memory 中进行的
// block_reduce (normal reduce)





// kernel2: 
// 上述对input的读取都是在 global memory 中进行的
// block_reduce (based warp reduce)

int main()
{
	// cpu:
	float* input, * output, * gpu_output;

	size_t input_bytes = sizeof(float) * M * N;
	size_t output_bytes = sizeof(float) * M * N;

	input = (float*)malloc(input_bytes);
	output = (float*)malloc(output_bytes);
	gpu_output = (float*)malloc(output_bytes);

	for (unsigned i = 0; i < M; ++i)
	{
		for(unsigned j=0; j<N; ++j)
			input[i*N+j] = static_cast<float>(rand()) / RAND_MAX;
	}

	cpu_softmax(input, output, M, N);
	//float cpu_sum = 0.0;
	//for (unsigned i = 0; i < N; ++i)
	//{
	//	if(i<50)
	//		printf("i: %d, input: %.8f, output: %.8f \n", i, input[i], output[i]);
	//	cpu_sum += output[i];
	//}
	//printf("cpu sum: %.2f \n", cpu_sum);


	// gpu:
	float* d_input, *d_output;

	cudaMalloc((void**)&d_input, input_bytes);
	cudaMalloc((void**)&d_output, output_bytes);

	cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

	unsigned grid_x = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 grid_size(grid_x);
	dim3 block_size(BLOCK_SIZE);
	softmax_thread<<<grid_size, block_size>>>(d_input, d_output, M, N);

	cudaMemcpy(gpu_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

	//// check:
	bool error = false;
	for (unsigned i = 0; i < M; ++i)
	{
		for (unsigned j = 0; j < N; ++j)
		{
			if (fabs(gpu_output[i*N+j] - output[i*N+j]) > 1e-8)
			{
				error = true;
			}

		}

	}

	printf("result: %s \n", error ? "fail" : "pass");

	int offset = 20;
	for (unsigned i = 0; i < offset + 20; ++i)
	{
		printf("gpu: %.8f, cpu: %.8f \n", gpu_output[i], output[i]);
	}

	// free:
	free(input);
	free(output);
	free(gpu_output);

	cudaFree(d_input);
	cudaFree(d_output);

	return 0;


}